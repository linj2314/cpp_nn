#include "main.hpp"

using namespace std;
using namespace cv;

void init(vector<pair<int, vector<int>>> & test) {
    //reading in all pixel data from training jpg files
    vector<string> simpson_characters = {"bart_simpson", 
                                        "charles_montgomery_burns", 
                                        "homer_simpson", 
                                        "krusty_the_clown", 
                                        "lisa_simpson", 
                                        "marge_simpson", 
                                        "milhouse_van_houten", 
                                        "moe_szyslak", 
                                        "ned_flanders", 
                                        "principal_skinner"};
    test.reserve(8000);
    for (int i = 0; i < 10; i++) {
        string cmd = "cd train && cd " + simpson_characters[i] + " && ls > ../../temp.txt";
        system(cmd.c_str());

        ifstream in = ifstream("temp.txt");
        string file_name;
        while (in >> file_name) {
            vector<int> v(784, 0);
            Mat image = imread("train/" + simpson_characters[i] + "/" + file_name, cv::IMREAD_GRAYSCALE);
            
            for (int j = 0; j < image.rows; j++) {
                for (int k = 0; k < image.cols; k++) {
                    v[j * 28 + k] = static_cast<int>(image.at<uchar>(j, k));
                }
            }
            test.push_back(make_pair(i, v));
        }
    }

    //shuffle training data so we can separate into batches
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine rng(seed);
    shuffle(test.begin(), test.end(), rng);
};

int main() {
    vector<pair<int, vector<int>>> test;

    init(test);

    return 0;
}