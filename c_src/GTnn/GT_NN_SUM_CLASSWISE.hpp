#include <immintrin.h>
#include <Eigen/Dense>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;
using namespace Eigen;

namespace Eigen { 
    typedef Eigen::Matrix<double, Eigen::Dynamic, 
                        Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor_t;
};

// DECLARATION OF CLASS GroupTestingNN BEGINS

namespace GT{
    class GroupTestingNN {
    public:
        GroupTestingNN(string &D_path, string &Q_path, 
                        string &R_path, string &D_name, 
                        unsigned int N, unsigned int Nq, unsigned int dim, 
                        double rho);
        bool load_data();
        void build_index();
        bool save_index(const string &filename);
        bool load_index(const string &filename);
        void search(); // 0 means one batch all queries
        void exhaustive_search();
        void save_results(); 
    
    protected:
        unsigned int dimention;
        unsigned int num_data;
        unsigned int num_query;
        unsigned int query_index;
        MatrixXdRowMajor_t data_set;
        MatrixXdRowMajor_t query_set;
        MatrixXdRowMajor_t data_cum;
        class unordered_map<unsigned int, unsigned int> index_map;

        std::string data_path;
        std::string query_path;
        std::string result_path;
        std::string dataset_name;
        class vector<vector<int>> search_res; 
        class vector<vector<int>> naive_res; 
        double threshold;

        unsigned long int num_dot_products = 0;
        double net_search_time = 0.0;
        double net_naive_time = 0.0;
        double mean_precision = 0.0;
        double mean_recall = 0.0;
        const string algo_name = "GroupTestingSumClasswise";

        static string path_append(const string& p1, const string& p2);
        void recursive_mkdir(const char *dir);
        bool extract_matrix(string filename, MatrixXdRowMajor_t &matrix, 
                            bool transpose, int max_rows);
        void save_matrix(string filename, MatrixXdRowMajor_t& matrix, 
                        unsigned int r, unsigned int c);
        void search_subspans(unsigned int start_data, unsigned int end_data, 
                                    double full_sim);
        void precision_and_recall();
        void check_file(ofstream &file);
    };
};

// DECLARATION OF CLASS GroupTestingNN ENDS

// IMPLEMENTATION OF PUBLIC METHODS BEGINS

GT::GroupTestingNN::GroupTestingNN(string &D_path, string &Q_path, 
                                    string &R_path, string &D_name, 
                                    unsigned int N, unsigned int Nq, 
                                    unsigned int dim, double rho) {
    this->data_path = D_path;
    this->query_path = Q_path;
    this->dataset_name = D_name;
    
    this->num_data = N;
    this->num_query = Nq;
    this->dimention = dim;
    this->threshold = rho;

    this->result_path = this->path_append(this->path_append(R_path, 
                                            this->algo_name),
                                D_name + "_rho" + to_string(rho));
    struct stat st = {0};
    if (stat((this->result_path).c_str(), &st) == -1) {
        recursive_mkdir((this->result_path).c_str());
    }
}

bool GT::GroupTestingNN::load_data() {
    this->data_set.resize(this->num_data, this->dimention);
    this->query_set.resize(this->num_query, this->dimention);

    bool is_loaded = this->extract_matrix(
        this->data_path, this->data_set, false, this->num_data);
    is_loaded = is_loaded && this->extract_matrix(
        this->query_path, this->query_set, false, this->num_query);
    return is_loaded;
}

void GT::GroupTestingNN::build_index() {
    class vector<vector<unsigned int>> class_wise_data;
    class_wise_data.resize(this->dimention);
    for (unsigned int i = 0; i < this->num_data; i++) {
        int max_in = -1;
        double max_val = 0.0;
        for (unsigned int j = 0; j < this->dimention; j++) {
            if (this->data_set(i, j) < max_val) continue;
            max_in = j;
            max_val = this->data_set(i, j);
        }
        class_wise_data[max_in].push_back(i);
    }
    this->data_cum.resize(this->num_data, this->dimention);
    unsigned int c_in = -1;
    for (unsigned int i = 0; i < this->dimention; i++) {
        for (unsigned int index : class_wise_data[i]) {
            this->index_map[++c_in] = index;
            this->data_cum.row(c_in) = this->data_set.row(index);
            if (c_in == 0) continue;
            this->data_cum.row(c_in) += this->data_cum.row(c_in - 1);
        }
    }
}

bool GT::GroupTestingNN::save_index(const string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return false; 
    }
    for (unsigned int i = 0; i < this->num_data; i++) {
        for (unsigned int j = 0; j < this->dimention; j++) {
            file << this->data_cum(i, j);
            if (j < (this->dimention - 1)) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Matrix successfully saved to " << filename << std::endl;
    return true;
}

bool GT::GroupTestingNN::load_index(const string &filename) {
    this->data_cum.resize(this->num_data, this->dimention);
    return this->extract_matrix(filename, this->data_cum, false, 
                                    this->num_data);
}

void GT::GroupTestingNN::search() {
    // std::cout << "Starting search" << std::endl;
    this->search_res.resize(this->num_query, vector<int>());
    this->net_search_time = 0.0;
    for (this->query_index = 0; this->query_index < this->num_query; 
                                    this->query_index++) {
        // std::cout << "Query index : " << this->query_index << std::endl;
        auto start = high_resolution_clock::now();
        double full_sim = this->query_set.row(this->query_index).dot(
            this->data_cum.row(this->num_data - 1));
        this->num_dot_products++;
        this->search_subspans(0, this->num_data - 1, full_sim);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(stop - start);
        this->net_search_time += duration.count() / 1.0e+6;
    }
    // std::cout << "Finished search" << std::endl;
}

void GT::GroupTestingNN::exhaustive_search() {
    this->naive_res.resize(this->num_query, vector<int>());
    double sim = 0;
    this->net_naive_time = 0.0;
    for (unsigned int i = 0; i < this->num_query; i++) {
        std::cout << "Query index : " << i << std::endl;
        auto start = high_resolution_clock::now();
        for (unsigned int row = 0; row < this->num_data; row++) {
            sim = this->data_set.row(row).dot(this->query_set.row(i));
            if (sim >= this->threshold)
                this->naive_res[i].push_back(row);
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(stop - start);
        this->net_naive_time += duration.count() / 1.0e+6;
    }
}

void GT::GroupTestingNN::save_results()
{
    this->precision_and_recall();
    std::ofstream fresults;
    std::ofstream faggregates;
    std::ofstream fgroundtruth;
    fresults.open(this->path_append(this->result_path, 
                                    string("results.txt")));
    check_file(fresults);
    fgroundtruth.open(this->path_append(this->result_path, 
                                        string("ground_truth.txt")));
    check_file(fgroundtruth);
    faggregates.open(this->path_append(this->result_path, string("agg.txt")),
                        ios::app);
    check_file(faggregates);
    // Fill in Search Results
    for (unsigned int i = 0; i < this->num_query; i++) {   
        for (unsigned int p = 0;  p < this->search_res[i].size(); p++) {
            if (p != this->search_res[i].size()-1) 
                fresults << this->search_res[i][p] << "," ;
            else 
                fresults << this->search_res[i][p] << std::endl;
        }
    }
    for (unsigned int i = 0; i < this->num_query; i++) {   
        for(unsigned int p = 0; p < this->naive_res[i].size(); p++) {
            if (p != this->naive_res[i].size() - 1) 
                fgroundtruth << this->naive_res[i][p] << "," ;
            else 
                fgroundtruth << this->naive_res[i][p] << std::endl;
        }
    } 
    // Dumping aggregates
    faggregates << "Algorithm : " << this->algo_name << std::endl;
    faggregates << "Dataset : " << this->dataset_name << std::endl;
    faggregates << "Avgerage pool search query time : " << 
                        this->net_search_time / this->num_query << std::endl;
    faggregates << "Avgerage exhaustive search query time : " << 
                        this->net_naive_time / this->num_query << std::endl;
    faggregates << "Average precision : " << this->mean_precision << std::endl;
    faggregates << "Average recall : " << this->mean_recall << std::endl;
    faggregates << "Number of dot products : " << 
                        this->num_dot_products << std::endl;
    // Safe close
    fresults.close();
    faggregates.close();
    fgroundtruth.close();
} 

// END OF IMPLEMENTATION OF PUBLIC METHODS

// IMPLEMENTATION OF PROTECTED METHODS BEGINS 

string GT::GroupTestingNN::path_append(const string& p1, const string& p2) {
    char sep = '/';
    std::string tmp = p1;

    #ifdef _WIN32
        sep = '\\';
    #endif

    if (p1[p1.length()] != sep) { 
        tmp += sep;
        return(tmp + p2);
    }
    else {
        return(p1 + p2);
    }
}

void GT::GroupTestingNN::recursive_mkdir(const char *dir) {
    char tmp[256];
    char *p = NULL;
    unsigned int len;
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

bool GT::GroupTestingNN::extract_matrix(string filename, 
                                        MatrixXdRowMajor_t &matrix, 
                                        bool transpose = false, 
                                        int max_rows = -1) {
    std::ifstream file(filename.c_str());
    std::string line;
    int rnum = 0, cnum = 0;
    if(file.fail()) {
        cout << "Couldn't open file : " << filename << std::endl;
        return false;
    } 
    while (getline(file, line))
    {
        if (!transpose && max_rows > 0 && rnum == max_rows) 
            break;
        std::istringstream level(line);
        std::string value;
        cnum = 0;
        while (getline(level, value, ','))
        {
            if (transpose && max_rows > 0 && cnum == max_rows) 
                break;
            double num = stod(value);
            if (transpose) matrix(cnum, rnum) = num;
            else matrix(rnum, cnum) = num;
            cnum++;
        }
        rnum++;
    }
    file.close();
    return true;
}

void GT::GroupTestingNN::save_matrix(string filename, 
                                        MatrixXdRowMajor_t& matrix, 
                                        unsigned int r, unsigned int c) {
    std::ofstream file;
    file.open(filename.c_str());
    if(file.fail()) {
        cout << "Couldn't open file : " << filename << endl;
        exit(1);
    } 
    for(unsigned int i = 0; i < r; i++) {
        for(unsigned int j = 0; j < c; j++) {
            if(j == c-1) file << matrix(i, j) << endl;
            else file << matrix(i, j) << ",";
        }
    }
}

void GT::GroupTestingNN::search_subspans(unsigned int start_data,
                                            unsigned int end_data, 
                                            double full_sim) {
    if (full_sim < this->threshold) return;
    if (start_data == end_data) {
        this->search_res[this->query_index].push_back(
            this->index_map[start_data]);
        return;
    }
    unsigned int mid_data = (start_data + end_data + 1) / 2;
    double half_sim = this->query_set.row(this->query_index).dot(
            this->data_cum.row(end_data) - this->data_cum.row(mid_data - 1));
    this->num_dot_products++;
    this->search_subspans(mid_data, end_data, half_sim);
    this->search_subspans(start_data, mid_data - 1,
                            full_sim - half_sim);
    return;
}

void GT::GroupTestingNN::precision_and_recall() {
    std::cout << "Calculating precision and recall " << std::endl;
    this->mean_precision = 0;
    this->mean_recall = 0;

    for(unsigned int i = 0; i < this->naive_res.size(); i++)
    {
        class unordered_set<int> truth_set(this->naive_res[i].begin(), 
                                        this->naive_res[i].end());
        unsigned int true_positives = 0;

        for (const auto& pred : this->search_res[i]) {
            if (truth_set.find(pred) != truth_set.end()) {
                true_positives++;
            }
        }
        this->mean_precision += (this->search_res[i].size() == 0) ? 1.0 :
            (static_cast<double>(true_positives) / this->search_res[i].size());
        this->mean_recall += (this->naive_res[i].size() == 0) ? 1.0 :
            (static_cast<double>(true_positives) / this->naive_res[i].size());

    }
    this->mean_precision /= this->search_res.size();
    this->mean_recall /= this->naive_res.size();
    return ;
}
 
void GT::GroupTestingNN::check_file(ofstream &file) {
    if(file.fail()) {
        std::cerr << "Unable to open file" << std::endl;
        exit(1);
    }
}

// END OF IMPLEMENTATION OF PROTECTED METHODS