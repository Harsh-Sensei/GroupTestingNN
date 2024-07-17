#include <immintrin.h>
#include <Eigen/Dense>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using json = nlohmann::json;

namespace Eigen { 
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor;
}

namespace GT {
    class GroupTestingNN {
        private:
            unsigned int dim, N, Nq;
            MatrixXdRowMajor X;
            MatrixXdRowMajor Q;
            MatrixXdRowMajor X_cum;
            MatrixXdRowMajor Q_cum;

            string data_path, query_path, result_path;
            float rho;

            // analysis variable 
            // vector<int> negPools;
            unsigned long int ntests;
            double agg_qtime = 0;
            double agg_gt_qtime = 0;
            double mean_precision = 0;
            double mean_recall = 0;

        public:
            const string algo_name = "DoubleGroupTestingSumEigen";
            
            class vector<vector<int>> search_res; 
            class vector<vector<int>> gt_res; 

            static string pathAppend(const string& p1, const string& p2);
            bool extractMatrix(string filename, MatrixXdRowMajor &X, bool transpose, int max_rows);
            void saveMatrix(string filename, MatrixXdRowMajor& mat, unsigned int r, unsigned int c);
            void recursive_mkdir(const char *dir);
            void createIndex(MatrixXdRowMajor& X_cum, MatrixXdRowMajor& X, int N);
            void precision_and_recall();
            void check_file(ofstream &file);

            GroupTestingNN(string &data_path, string &query_path, string &result_path, string &dname, 
                unsigned int N, unsigned int Nq, unsigned int dim, float rho);
            bool load_data();
            bool build_index();
            bool save_index(const string &filename);
            bool load_index(const string &filename);
            void search_subspans(int start_data, int end_data, int start_query, int end_query, double full_sim);
            void search(int batch_size = 0); // 0 means one batch all queries
            void search(MatrixXdRowMajor &Q, int batch_size = 0);
            void exhaustive_search();
            void exhaustive_search(MatrixXdRowMajor &Q);
            void save_results(); 
    };
};

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
 
bool GT::GroupTestingNN::extractMatrix(
    string filename, MatrixXdRowMajor &X, bool transpose = false, int max_rows = -1) {
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

void GT::GroupTestingNN::saveMatrix(
    string filename, MatrixXdRowMajor& mat, unsigned int r, unsigned int c){
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
 
void GT::GroupTestingNN::createIndex(MatrixXdRowMajor& X_cum, MatrixXdRowMajor& X, int N) {
    X_cum.row(0) = X.row(0);
    for (int i = 1; i < N; i++)
    {
        X_cum.row(i) = X_cum.row(i-1) + X.row(i);
    }
}

void GT::GroupTestingNN::precision_and_recall() {
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
 
void GT::GroupTestingNN::check_file(ofstream &file) {
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

GT::GroupTestingNN::GroupTestingNN(
    string &data_path, string &query_path, string &result_path, string &dname, 
    unsigned int N, unsigned int Nq, unsigned int dim, float rho) {
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

bool GT::GroupTestingNN::load_data() {
    this->X.resize(this->N, this->dim);
    this->Q.resize(this->Nq, this->dim);

    bool is_loaded = this->extractMatrix(this->data_path, this->X, false, this->N);
    is_loaded = is_loaded && this->extractMatrix(this->query_path, this->Q, false, this->Nq);
    return is_loaded;
}

bool GT::GroupTestingNN::build_index() {
    this->X_cum.resize(this->N, this->dim);
    this->createIndex(this->X_cum, this->X, N);
    this->Q_cum.resize(this->Nq, this->dim);
    this->createIndex(this->Q_cum, this->Q, Nq);
    return true; 
}

bool GT::GroupTestingNN::save_index(const string &filename) {
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

bool GT::GroupTestingNN::load_index(const string &filename) {
    this->X_cum.resize(this->N, this->dim);
    return this->extractMatrix(filename, this->X_cum, false, this->N);
}

void GT::GroupTestingNN::search_subspans(
    int start_data, int end_data, int start_query, int end_query, double full_sim) {
    if (full_sim < rho) {
        return;
    }
    if (start_data == end_data && start_query == end_query) {
        this->search_res[start_query].push_back(start_data);
        return;
    }
    this->ntests++;
    int mid_data = (start_data + end_data + 1) / 2;
    int mid_query = (start_query + end_query + 1) / 2;

    if (start_data == end_data) 
    {
        double half_sim = this->X_cum.row(start_data).dot(this->Q_cum.row(end_query));
        half_sim -= this->X_cum.row(start_data).dot(this->Q_cum.row(mid_query-1));
        half_sim -= (start_data > 0) ? this->X_cum.row(start_data-1).dot(this->Q_cum.row(end_query)) : 0;
        half_sim += (start_data > 0) ? this->X_cum.row(start_data-1).dot(this->Q_cum.row(mid_query-1)) : 0;

        this->search_subspans(start_data, end_data, mid_query, end_query, half_sim);
        this->search_subspans(start_data, end_data, start_query, mid_query-1, full_sim - half_sim);
    }
    if (start_query == end_query) 
    {
        double half_sim = this->X_cum.row(end_data).dot(this->Q_cum.row(start_query));
        half_sim -= this->X_cum.row(mid_data-1).dot(this->Q_cum.row(start_query));
        half_sim -= (start_query > 0) ? this->X_cum.row(end_data).dot(this->Q_cum.row(start_query-1)) : 0;
        half_sim += (start_query > 0) ? this->X_cum.row(mid_data-1).dot(this->Q_cum.row(start_query-1)) : 0;

        this->search_subspans(mid_data, end_data, start_query, end_query, half_sim);
        this->search_subspans(start_data, mid_data-1, start_query, end_query, full_sim - half_sim);
    }
    
    double third_sim = this->X_cum.row(end_data).dot(this->Q_cum.row(end_query));
    third_sim -= this->X_cum.row(mid_data-1).dot(this->Q_cum.row(end_query));
    third_sim -= this->X_cum.row(end_data).dot(this->Q_cum.row(mid_query-1));
    third_sim += this->X_cum.row(mid_data-1).dot(this->Q_cum.row(mid_query-1));

    double half_sim = this->X_cum.row(mid_data-1).dot(this->Q_cum.row(end_query));
    half_sim -= this->X_cum.row(mid_data-1).dot(this->Q_cum.row(mid_query-1));
    half_sim -= (start_data > 0) ? this->X_cum.row(start_data-1).dot(this->Q_cum.row(end_query)) : 0;
    half_sim += (start_data > 0 && start_query > 0) ? this->X_cum.row(start_data-1).dot(this->Q_cum.row(start_query-1)) : 0;

    double quart_sim = this->X_cum.row(mid_data-1).dot(this->Q_cum.row(end_query));
    quart_sim -= this->X_cum.row(mid_data-1).dot(this->Q_cum.row(mid_query-1));
    quart_sim -= (start_data > 0) ? this->X_cum.row(start_data-1).dot(this->Q_cum.row(end_query)) : 0;
    quart_sim += (start_data > 0) ? this->X_cum.row(start_data-1).dot(this->Q_cum.row(mid_query-1)) : 0;

    this->search_subspans(mid_data, end_data, mid_query, end_query, third_sim);
    this->search_subspans(start_data, mid_data-1, mid_query, end_query, quart_sim);
    this->search_subspans(mid_data, end_data, start_query, mid_query-1, full_sim - half_sim - third_sim);
    this->search_subspans(start_data, mid_data-1, start_query, mid_query-1, half_sim - quart_sim);
}

void GT::GroupTestingNN::search(int batch_size = 0) {
    if (batch_size == 0) batch_size = this->Nq;
    this->search_res.resize(this->Nq, vector<int>());
    this->ntests = 0;

    for (int sq = 0; sq < this->Nq; sq += batch_size) {
        auto start = high_resolution_clock::now();
        int eq = min(sq + batch_size - 1, this->Nq - 1);
        this->search_subspans(0, N-1, sq, eq, 
                this->X_cum.row(this->N-1).dot(this->Q_cum.row(eq)) -
                (sq > 0) ? this->X_cum.row(this->N-1).dot(this->Q_cum.row(sq-1)) : 0);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        this->agg_qtime += duration.count();
    }
}

void GT::GroupTestingNN::search(MatrixXdRowMajor &Q, int batch_size = 0) {
    int Nq = Q.rows();
    if (batch_size == 0) batch_size = Nq;
    this->search_res.resize(Nq, vector<int>());
    this->createIndex(this->Q_cum, Q, Nq);
    this->ntests = 0;

    for (int sq = 0; sq < Nq; sq += batch_size) {
        auto start = high_resolution_clock::now();
        int eq = min(sq + batch_size - 1, Nq - 1);
        this->search_subspans(0, N-1, sq, eq, 
                this->X_cum.row(this->N-1).dot(Q.row(eq)) -
                (sq > 0) ? this->X_cum.row(this->N-1).dot(Q.row(sq-1)) : 0);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        this->agg_qtime += duration.count();
    }
    this->createIndex(this->Q_cum, this->Q, this->Nq); 
    // restore the original query. comment above if not needed.
}

void GT::GroupTestingNN::exhaustive_search() {
    this->gt_res.resize(this->Nq, vector<int>());
    double sim = 0;
    for (unsigned int i = 0; i < this->Nq; i++)
    {
        cout << "Query idx : " << i << endl;
        auto start = high_resolution_clock::now();
        for (unsigned int row = 0; row < this->N; row++)
        {
            sim = X.row(row).dot(this->Q.row(i));
            if (sim >= this->rho)
            {
                this->gt_res[i].push_back(row);
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        agg_gt_qtime += duration.count();
    }
}

void GT::GroupTestingNN::exhaustive_search(MatrixXdRowMajor &Q) {
    int Nq = Q.rows();
    this->gt_res.resize(Nq, vector<int>());
    double sim = 0;
    for (unsigned int i = 0; i < Nq; i++)
    {
        cout << "Query idx : " << i << endl;
        auto start = high_resolution_clock::now();
        for (unsigned int row = 0; row < this->N; row++)
        {
            sim = X.row(row).dot(Q.row(i));
            if (sim >= this->rho)
            {
                this->gt_res[i].push_back(row);
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        agg_gt_qtime += duration.count();
    }
}

void GT::GroupTestingNN::save_results()
{
    this->precision_and_recall();
    ofstream fresults, faggregates, fgroundtruth;
    fresults.open(this->pathAppend(this->result_path, string("results.txt")));
    check_file(fresults);

    fgroundtruth.open(this->pathAppend(this->result_path, string("ground_truth.txt")));
    check_file(fgroundtruth);
    
    faggregates.open(this->pathAppend(this->result_path, string("agg.txt")));
    check_file(faggregates);

    for (unsigned int i = 0; i < this->Nq; i++)
    {   
        for(unsigned int p=0;  p < this->search_res[i].size(); p++)
        {
            if (p != this->search_res[i].size()-1) fresults << this->search_res[i][p] << " , " ;
            else fresults << this->search_res[i][p] << endl;
        }
    }

    for (unsigned int i = 0; i < this->Nq; i++)
    {   
        for(unsigned int p=0;  p < this->gt_res[i].size(); p++)
        {
            if (p != this->gt_res[i].size()-1) fgroundtruth << this->gt_res[i][p] << " , " ;
            else fgroundtruth << this->gt_res[i][p] << endl;
        }
    }

    // Dumping aggregates
    faggregates << "Algorithm : " << this->algo_name << endl;
    faggregates << "Avgerage pool search query time : " << this->agg_qtime/this->Nq << endl;
    faggregates << "Avgerage exhaustive search query time : " << this->agg_gt_qtime/this->Nq << endl;
    faggregates << "Average precision : " << this->mean_precision << endl;
    faggregates << "Average recall : " << this->mean_recall << endl;
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