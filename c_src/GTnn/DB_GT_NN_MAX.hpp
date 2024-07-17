#include <immintrin.h>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

namespace GT {
    struct node_t {
        class vector<double> max_vec;
        class vector<double> min_vec;
        node_t* left = nullptr;
        node_t* right = nullptr;
        int start_index = 0;
        int size = 0;

        node_t();
        node_t(vector<double> &vec, int start);
        node_t(vector<double> &max_vec, vector<double> &min_vec, int start);
        node_t(node_t* left, node_t* right);
        ~node_t();
    };

    class GroupTestingNN {
        unsigned int dim, N, Nq;
            class vector<vector<double>> X;
            class vector<vector<double>> Q;
            node_t* X_tree = nullptr;
            node_t* Q_tree = nullptr;

            unsigned int index_sz;
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
            const string algo_name = "DoubleGroupTestingSignedMax";
            
            class vector<vector<int>> search_res; 
            class vector<vector<int>> gt_res; 

            static string pathAppend(const string& p1, const string& p2);
            bool extractMatrix(string filename, vector<vector<double>> &X, bool transpose, int max_rows);
            void saveMatrix(string filename, vector<vector<double>> &mat);
            void recursive_mkdir(const char *dir);
            node_t* build_tree(vector<vector<double>> &X, int l, int r);
            void precision_and_recall();
            void check_file(ofstream &file);

            GroupTestingNN(string &data_path, string &query_path, string &result_path, string &dname, 
                unsigned int N, unsigned int Nq, unsigned int dim, float rho);
            ~GroupTestingNN();
            inline int find_index(int round, int branch);

            bool load_data();
            bool build_index();
            bool save_index(const string &filename);
            bool load_index(const string &filename);
            void search_subtree(node_t* data, node_t* query);
            void search();
            void search(vector<vector<double>>& Q);
            void exhaustive_search();
            void exhaustive_search(vector<vector<double>>& Q);
            void save_results();
    };
};

GT::node_t::node_t() {
    left = nullptr;
    right = nullptr;
    start_index = 0;
    size = 0;
}

GT::node_t::node_t(vector<double> &vec, int start = 0) {
    max_vec = vec;
    min_vec = vec;
    left = nullptr;
    right = nullptr;
    start_index = start;
    size = 1;
}

GT::node_t::node_t(vector<double> &max_vec, vector<double> &min_vec, int start = 0) {
    this->max_vec = max_vec;
    this->min_vec = min_vec;
    left = nullptr;
    right = nullptr;
    start_index = start;
    size = 1;
}

GT::node_t::node_t(GT::node_t* left, GT::node_t* right) {
    this->left = left;
    this->right = right;
    size = 0;
    if (left) {
        max_vec.resize(left->max_vec.size(), INT_MIN);
        min_vec.resize(left->max_vec.size(), INT_MAX);
        start_index = left->start_index;
        size += left->size;
        for (int i = 0; i < left->max_vec.size(); i++) {
            max_vec[i] = max(max_vec[i], left->max_vec[i]);
            min_vec[i] = min(min_vec[i], left->min_vec[i]);
        }
    }
    if (right) {
        if (left && left->max_vec.size() != right->max_vec.size()) {
            throw invalid_argument("left and right node have different dimensions");
        }
        if (!left) {
            max_vec.resize(right->max_vec.size(), INT_MIN);
            min_vec.resize(right->max_vec.size(), INT_MAX);
            start_index = right->start_index;
        }
        size += right->size;
        for (int i = 0; i < right->max_vec.size(); i++) {
            max_vec[i] = max(max_vec[i], right->max_vec[i]);
            min_vec[i] = min(min_vec[i], right->min_vec[i]);
        }
    }
}

GT::node_t::~node_t() {
    if (left) delete left;
    if (right) delete right;
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
 
bool GT::GroupTestingNN::extractMatrix(string filename, vector<vector<double>> &X, 
    bool transpose = false, int max_rows = -1) {
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
                X[cnum][rnum] = num;
            else
                X[rnum][cnum] = num;

            cnum++;
        }
        rnum++;
    }
    file.close();

    return true;
}

 
void GT::GroupTestingNN::saveMatrix(string filename, vector<vector<double>> &mat) {
    ofstream file;
    file.open(filename.c_str());
    if(file.fail())
    {
        cout << "Couldn't open file : " << filename << endl;
        exit(1);
    } 
    for(unsigned int i = 0; i < mat.size(); i++)
    {
        for(unsigned int j = 0; j < mat[i].size(); j++)
        {
            if(j == c-1)
                file << mat[i][j] << endl;
            else
                file << mat[i][j] << " , ";
        }
    }

    return;
}

GT::node_t* GT::GroupTestingNN::build_tree(vector<vector<double>> &X, int l, int r) {
    if (l == r) {
        return new node_t(X[l]);
    }
    int m = (l + r) / 2;
    node_t* left = build_tree(l, m);
    node_t* right = build_tree(m + 1, r);
    return new node_t(left, right);
}

void GT::GroupTestingNN::precision_and_recall() {
    mean_precision = 0;
    mean_recall = 0;

    for(unsigned int i=0; i<this->Nq; i++)
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
    mean_precision /= this->Nq;
    mean_recall /= this->Nq;
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

GT::GroupTestingNN::~GroupTestingNN() {
    if (X_tree) delete X_tree;
    if (Q_tree) delete Q_tree;
}

bool GT::GroupTestingNN::load_data() {
    this->X.resize(this->N, vector<double>(this->dim));
    this->Q.resize(this->Nq, vector<double>(this->dim));
    bool is_loaded = this->extractMatrix(this->data_path, this->X, false, this->N);
    is_loaded = is_loaded && this->extractMatrix(this->query_path, this->Q, false, this->Nq);
    return is_loaded;
}

bool GT::GroupTestingNN::build_index() {
    this->X_tree = build_tree(this->X, 0, this->N - 1);
    this->Q_tree = build_tree(this->Q, 0, this->Nq - 1);
    return true;
}

void GT::GroupTestingNN::search_subtree(node_t* data, node_t* query) {
    if (data == nullptr || query == nullptr) {
        throw invalid_argument("data or query node is null");
    }
    double dot = 0;
    this->ntests++;
    for (int i = 0; i < data->max_vec.size(); i++) {
        dot += max(data->min_vec[i] * query->min_vec[i], data->max_vec[i] * query->max_vec[i]);
    }
    if (dot < this->rho) {
        return;
    }
    if (data->size == 1 && query->size == 1) {
        this->search_res[query->start_index].push_back(data->start_index);
        return;
    }
    if (data->size == 1) {
        search_subtree(data, query->left);
        search_subtree(data, query->right);
        return;
    }
    if (query->size == 1) {
        search_subtree(data->left, query);
        search_subtree(data->right, query);
        return;
    }
    search_subtree(data->left, query->left);
    search_subtree(data->left, query->right);
    search_subtree(data->right, query->left);
    search_subtree(data->right, query->right);
}

void GT::GroupTestingNN::search() {
    auto start = high_resolution_clock::now();
    search_subtree(this->X_tree, this->Q_tree);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    this->agg_qtime += duration.count();
}

void GT::GroupTestingNN::search(vector<vector<double>>& Q) {
    this->Q_tree = build_tree(Q, 0, Q.size() - 1);
    auto start = high_resolution_clock::now();
    search_subtree(this->X_tree, this->Q_tree);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    this->agg_qtime += duration.count();
    this->Q_tree = build_tree(this->Q, 0, this->Nq - 1);
    // Above line is to reset the Q_tree to original query set
}

void GT::GroupTestingNN::exhaustive_search() {
    this->gt_res.resize(this->Nq, vector<int>());
    double sim = 0;
    for (unsigned int i = 0; i < this->Nq; i++)
    {
        cout << "Query idx : " << i << endl;
        auto start = high_resolution_clock::now();
        for (unsigned int row = 0; row <this->N; row++)
        {
            sim = 0;
            for(unsigned int e=0; e<this->dim; e++) sim += (this->X[row][e])*(this->Q[i][e]);
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

void GT::GroupTestingNN::exhaustive_search(vector<vector<double>>& Q) {
    this->gt_res.resize(this->Nq, vector<int>());
    double sim = 0;
    for (unsigned int i = 0; i < this->Nq; i++)
    {
        cout << "Query idx : " << i << endl;
        auto start = high_resolution_clock::now();
        for (unsigned int row = 0; row <this->N; row++)
        {
            sim = 0;
            for(unsigned int e=0; e<this->dim; e++) sim += (this->X[row][e])*(Q[i][e]);
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

void GT::GroupTestingNN::save_results() {
    ofstream file;
    string filename = this->pathAppend(this->result_path, "results.json");
    file.open(filename.c_str());
    check_file(file);
    json j;
    j["algo_name"] = this->algo_name;
    j["rho"] = this->rho;
    j["N"] = this->N;
    j["Nq"] = this->Nq;
    j["dim"] = this->dim;
    j["agg_qtime"] = this->agg_qtime;
    j["agg_gt_qtime"] = this->agg_gt_qtime;
    j["mean_precision"] = this->mean_precision;
    j["mean_recall"] = this->mean_recall;
    j["ntests"] = this->ntests;
    file << j.dump(4) << endl;
    file.close();

    filename = this->pathAppend(this->result_path, "search_res.csv");
    file.open(filename.c_str());
    check_file(file);
    for (unsigned int i = 0; i < this->Nq; i++) {
        for (unsigned int j = 0; j < this->search_res[i].size(); j++) {
            file << this->search_res[i][j];
            if (j < this->search_res[i].size() - 1) {
                file << ",";
            }
        }
        file << endl;
    }
    file.close();

    filename = this->pathAppend(this->result_path, "gt_res.csv");
    file.open(filename.c_str());
    check_file(file);
    for (unsigned int i = 0; i < this->Nq; i++) {
        for (unsigned int j = 0; j < this->gt_res[i].size(); j++) {
            file << this->gt_res[i][j];
            if (j < this->gt_res[i].size() - 1) {
                file << ",";
            }
        }
        file << endl;
    }
    file.close();
}