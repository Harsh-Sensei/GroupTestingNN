#include <immintrin.h>
#include <Eigen/Dense>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace std;
using namespace std::chrono;
using namespace Eigen;


namespace Eigen{ 
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor;
}


namespace GT{
    class GroupTestingNN{
        private:
            unsigned int dim, N, Nq;
            MatrixXdRowMajor X;
            MatrixXdRowMajor Q;
            MatrixXdRowMajor X_cum;

            string data_path, query_path, result_path;
            float rho;
            float init_frac = 0.8;
            int last_idx;

            // analysis variable 
            vector<int> negPools;
            unsigned long int ntests, num_insert, num_search;

            double agg_qtime = 0;
            double agg_gt_qtime = 0;
            double agg_insertion_time = 0;
            double index_build_time = 0;

            double mean_precision = 0;
            double mean_recall = 0;

        public:
            const string algo_name = "GroupTestingStreamingEigen";
            
            vector<vector<int>> search_res; 
            vector<vector<int>> gt_res; 

            static string pathAppend(const string& p1, const string& p2);
            bool extractMatrix(string filename, MatrixXdRowMajor &X, bool transpose, int max_rows);
            void saveMatrix(string filename, MatrixXdRowMajor& mat, unsigned int r, unsigned int c);
            void recursive_mkdir(const char *dir);
            void createIndex(MatrixXdRowMajor& X_cum, MatrixXdRowMajor& X, int N);
            void precision_and_recall();
            void check_file(ofstream &file);

            GroupTestingNN(string &data_path, string &query_path, string &result_path, string &dname, unsigned int N, unsigned int Nq, unsigned int dim, float rho);
            bool load_data();
            bool build_index(float init_frac);
            bool save_index(const string &filename);
            bool load_index(const string &filename);
            void search();
            void search(MatrixXdRowMajor &Q);
            void exhaustive_search();   
            void exhaustive_search(MatrixXdRowMajor &Q);  

            void simulate_streaming_exh(const string&);
            void simulate_streaming(const string&);

            void save_results();   
    };
}


string GT::GroupTestingNN::pathAppend(const string& p1, const string& p2) {

    char sep = '/';
    string tmp = p1;

#ifdef _WIN32
    sep = '\\';
#endif

    if (p1[p1.length()] != sep) 
    { 
        tmp += sep;
        return(tmp + p2);
    }
    else
        return(p1 + p2);
}
 
bool GT::GroupTestingNN::extractMatrix(string filename, MatrixXdRowMajor &X, bool transpose = false, int max_rows = -1)
{
    ifstream file;
    string line;
    int rnum = 0, cnum = 0;
    file.open(filename.c_str());
    if(file.fail())
    {
        cout << "Couldn't open file : " << filename << endl;
        return false;
    } 
    while (getline(file, line))
    {
        if(!transpose && max_rows > 0 && rnum == max_rows)
            break;
        istringstream level(line);

        string value;
        cnum = 0;
        while (getline(level, value, ','))
        {
            if(transpose && max_rows > 0 && cnum == max_rows)
                break;
            double num = stod(value);
            if(transpose)
                X(cnum, rnum) = num;
            else
                X(rnum, cnum) = num;

            cnum++;
        }
        rnum++;
    }
    file.close();

    return true;
}

 
void GT::GroupTestingNN::saveMatrix(string filename, MatrixXdRowMajor& mat, unsigned int r, unsigned int c)
{
    ofstream file;
    file.open(filename.c_str());
    if(file.fail())
    {
        cout << "Couldn't open file : " << filename << endl;
        exit(1);
    } 
    for(unsigned int i = 0; i < r; i++)
    {
        for(unsigned int j = 0; j < c; j++)
        {
            if(j == c-1)
                file << mat(i, j) << endl;
            else
                file << mat(i, j) << " , ";
        }
    }

    return;
}
 
void GT::GroupTestingNN::createIndex(MatrixXdRowMajor& X_cum, MatrixXdRowMajor& X, int N)
{
    X_cum.row(0) = X.row(0);
    for (int i = 1; i < N; i++)
    {
        X_cum.row(i) = X_cum.row(i-1) + X.row(i);
    }
}


void GT::GroupTestingNN::precision_and_recall()
{
    cout << "Calculating precision and recall " <<endl;
    mean_precision = 0;
    mean_recall = 0;

    for(unsigned int i=0; i<this->gt_res.size(); i++)
    {
        std::unordered_set<int> truthSet(this->gt_res[i].begin(), this->gt_res[i].end());
        int truePositives = 0;

        for (const auto& pred : this->search_res[i]) {
            if (truthSet.find(pred) != truthSet.end()) {
                truePositives++;
            }
        }
        mean_precision += this->search_res[i].size() > 0 ? (static_cast<double>(truePositives) / this->search_res[i].size()) : 1.0;
        mean_recall += this->gt_res[i].size() > 0 ? (static_cast<double>(truePositives) / this->gt_res[i].size()) : 1.0;

    }
    mean_precision /= this->gt_res.size();
    mean_recall /= this->gt_res.size();
    return ;
}
 
void GT::GroupTestingNN::check_file(ofstream &file)
{
    if(file.fail())
    {
        cout << "Unable to open file" << endl;
        exit(1);
    }
}
 
void GT::GroupTestingNN::recursive_mkdir(const char *dir) {
    char tmp[256];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp),"%s",dir);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    mkdir(tmp, S_IRWXU);
}


GT::GroupTestingNN::GroupTestingNN(string &data_path, string &query_path, string &result_path, string &dname, unsigned int N, unsigned int Nq, unsigned int dim, float rho)
{
    this->data_path = data_path;
    this->query_path = query_path;
    this->rho = rho;
    this->N = N;
    this->Nq = Nq;
    this->dim = dim;

    this->result_path = this->pathAppend(this->pathAppend(result_path, this->algo_name), dname + "_rho" + to_string(this->rho));
    struct stat st = {0};
    if (stat((this->result_path).c_str(), &st) == -1) {
        recursive_mkdir((this->result_path).c_str());
    }
}

bool GT::GroupTestingNN::load_data()
{
    this->X.resize(this->N, this->dim);
    this->Q.resize(this->Nq, this->dim);

    bool is_loaded = this->extractMatrix(this->data_path, this->X, false, this->N);
    is_loaded = is_loaded && this->extractMatrix(this->query_path, this->Q, false, this->Nq);
    return is_loaded;
}

bool GT::GroupTestingNN::build_index(float init_frac=0.8)
{
    auto start = high_resolution_clock::now();
    if (init_frac > 1)
    {
        throw std::runtime_error("init_frac > 1 not possible");
    }
    else if (init_frac < 1 && init_frac > 0)
    {
        this->last_idx = int(this->N*init_frac);
        this->init_frac = init_frac;
        this->X_cum.resize(this->N, this->dim);
        this->createIndex(this->X_cum, this->X, this->last_idx);
    }
    else
    {
        this->X_cum.resize(this->N, this->dim);
        this->createIndex(this->X_cum, this->X, this->N);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    this->index_build_time = duration.count();
    return true; 
}

bool GT::GroupTestingNN::save_index(const string &filename)
{
    std::ofstream file(filename);
    if (file.is_open()) {
        for (unsigned int i = 0; i < this->N; ++i) {
            for (unsigned int j = 0; j < this->dim; ++j) {
                file << this->X_cum(i, j);
                if (j < (this->dim - 1)) file << ",";
            }
            file << "\n";
        }
        file.close();
        std::cout << "Matrix successfully saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }

    return true;
}

bool GT::GroupTestingNN::load_index(const string &filename)
{
    this->X_cum.resize(this->N, this->dim);
    return this->extractMatrix(filename, this->X_cum, false, this->N);
}


void GT::GroupTestingNN::simulate_streaming(const string& filename)
{
    std::ifstream f(filename);
    json streaming_steps = json::parse(f);
    int num_steps = streaming_steps.size();

    this->num_insert = 0;
    this->num_search = 0;
    this->search_res.clear();

    int mxl = log2(this->N) + 1, level, si, ei, nele, round;
    double stk[4][mxl], sim;
    (this->negPools).resize(mxl, 0);
    this->ntests = 0;

    for (int i = 0; i < num_steps; i++)
    {
        cout << "Step number : " << i << "/" << num_steps << endl;
        string action = streaming_steps[to_string(i)]["action"];
        int value = streaming_steps[to_string(i)]["value"];

        if (action == "search")
        {
            cout << "Searching " << value << "th vector" << endl; 

            this->num_search += 1;
            this->search_res.push_back(vector<int>());

            // Running binary splitting method for each query with t as threshold.
            level = 0, si = 0, ei = this->last_idx - 1;
            auto start = high_resolution_clock::now();
            
            // Start
            sim = (this->X_cum).row(ei).dot(this->Q.row(value));
            this->ntests++;

            stk[0][0] = si;
            stk[1][0] = ei;
            stk[2][0] = sim;
            stk[3][0] = 0;
            while (level >= 0)
            {
                si = stk[0][level];
                ei = stk[1][level];
                sim = stk[2][level];
                round = stk[3][level];

                if (sim >= this->rho)
                {
                    nele = (ei - si + 1);

                    if (nele == 1)
                    {
                            this->search_res.back().push_back(si);
                            level -= 1;
                    }
                    else
                    {
                        int index = floor(nele / 2);

                        double rsim = (X_cum.row(ei) - X_cum.row(si + index - 1)).dot(this->Q.row(value));
                        this->ntests++;

                        stk[0][level] = si + index;
                        stk[1][level] = ei;
                        stk[2][level] = rsim;
                        stk[3][level] = round + 1;
                        level++;
                        stk[0][level] = si;
                        stk[1][level] = si + index - 1;
                        stk[2][level] = sim - rsim;
                        stk[3][level] = round + 1;
                    }
                }
                else
                {
                    this->negPools[round]++;
                    level--;
                }
            }
            // End
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - start);
            this->agg_qtime += duration.count();

        }
        if (action == "insert")
        {  
            cout << "Inserting " << value << " vectors into database of " << last_idx << "vectors" << endl; 
            if (last_idx + value > (int)this->N)
            {
                cout << "Cannot insert more data, skipping..." << endl;
                continue;
            }
            this->num_insert += 1;
            auto start_insert = high_resolution_clock::now();
            for(int _i = 0; _i < value; _i++)
            {
                if(_i + last_idx < (int)N)
                    X_cum.row(_i + last_idx) = X_cum.row(_i + last_idx - 1) + X.row(_i + last_idx);
            }
            auto stop_insert = high_resolution_clock::now();
            auto duration_insert = duration_cast<milliseconds>(stop_insert - start_insert);
            this->agg_insertion_time += duration_insert.count();
            last_idx += value;
        }
    }

}

void GT::GroupTestingNN::simulate_streaming_exh(const string& filename)
{
    std::ifstream f(filename);
    json streaming_steps = json::parse(f);
    int num_steps = streaming_steps.size();
    
    this->last_idx = int(this->N*this->init_frac);
    this->gt_res.clear();

    double sim = 0;
    for (int i = 0; i < num_steps; i++)
    {
        cout << "Step number : " << i << "/" << num_steps << endl;
        string action = streaming_steps[to_string(i)]["action"];
        int value = streaming_steps[to_string(i)]["value"];
        if (action == "search")
        {
            this->gt_res.push_back(vector<int>());
            auto start_exh = high_resolution_clock::now();

            for (int row = 0; row < this->last_idx; row++)
            {
                sim = X.row(row).dot(Q.row(value));
                if (sim >= this->rho)
                    this->gt_res.back().push_back(row);
            }
            auto stop_exh = high_resolution_clock::now();
            auto duration_exh = duration_cast<milliseconds>(stop_exh - start_exh);
            this->agg_gt_qtime += duration_exh.count();
        }
        if (action == "insert")
        {  
            if (last_idx + value > (int)this->N)
            {
                cout << "Cannot insert more data, skipping..." << endl;
                continue;
            }
            last_idx += value;
        }
    }

}

void GT::GroupTestingNN::save_results()
{
    precision_and_recall();
    ofstream fresults, faggregates, fgroundtruth;

    fresults.open(this->pathAppend(this->result_path, string("results.txt")));
    check_file(fresults);

    fgroundtruth.open(this->pathAppend(this->result_path, string("ground_truth.txt")));
    check_file(fgroundtruth);
    
    faggregates.open(this->pathAppend(this->result_path, string("agg.txt")));
    check_file(faggregates);

    for (unsigned int i = 0; i < this->search_res.size(); i++)
    {   
        for(unsigned int p=0;  p < this->search_res[i].size(); p++)
        {
            if (p != this->search_res[i].size()-1) fresults << this->search_res[i][p] << " , " ;
            else fresults << this->search_res[i][p] << endl;
        }
    }

    for (unsigned int i = 0; i < this->gt_res.size(); i++)
    {   
        for(unsigned int p=0;  p < this->gt_res[i].size(); p++)
        {
            if (p != this->gt_res[i].size()-1) fgroundtruth << this->gt_res[i][p] << " , " ;
            else fgroundtruth << this->gt_res[i][p] << endl;
        }
    }

    // Dumping aggregates
    faggregates << "Algorithm : " << this->algo_name << endl;
    faggregates << "Num search : " << this->num_search << endl;
    faggregates << "Num insert : " << this->num_insert << endl;
    faggregates << "Avgerage pool search query time : " << this->agg_qtime/this->num_search << endl;
    faggregates << "Avgerage exhaustive search query time : " << this->agg_gt_qtime/this->num_search << endl;
    faggregates << "Average precision : " << this->mean_precision << endl;
    faggregates << "Average recall : " << this->mean_recall << endl;
    faggregates << "Index generation time : " << this->index_build_time << endl;
    faggregates << "Avgerage insertion time grp: " << this->agg_insertion_time/this->num_insert << endl;
    faggregates << "Number of dot products : " << this->ntests << endl;
    faggregates << "Number of neg pools : " ;
    for(unsigned int i = 0 ; i < this->negPools.size(); i++)
    {
        if (i != this->negPools.size()-1) faggregates << this->negPools[i] << " , " ;
        else faggregates << this->negPools[i] << endl;
    }
    
    // Safe close
    fresults.close();
    faggregates.close();
    fgroundtruth.close();
}  

