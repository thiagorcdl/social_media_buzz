![preview](https://github.com/thiagorcdl/social_media_buzz/blob/master/assets/social_media_preview.png)

# Social Media Buzz
[![Released under the MIT license.](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/thiagorcdl/social_media_buzz/blob/master/LICENSE) [![This Repository uses a generated Social Preview from @pqt/social-preview](https://img.shields.io/badge/%E2%9C%93-Social%20Preview-blue)](https://github.com/pqt/social-preview)

### Ranking Significant Features for Increasing Engagement on Social Media via Regression Analysis.

This code was developed as a study tool for the Predictive Modeling, Model Fitting, and Regression Analysis course provided by the University of California Irvine on Coursera.
It utilizes the Buzz in Social Media data set, available at the UCI Machine Learning Repository, for identifying the attributes in social media content that have the highest correlation to the amount of repercussion it gained. To achieve such result, several linear regression models are constructed, then ranked based on their respective model fit measure (R-square).

# Usage

1. Clone repository
1. Fetch dataset ([regression.tar.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/00248/))
1. Extract inside `{PROJECT_ROOT}/assets/dataset` so you have the following directories:
    - `{PROJECT_ROOT}/assets/dataset/regression/Twitter`
    - `{PROJECT_ROOT}/assets/dataset/regression/TomsHardware` (won't be used)
1. Install requirements:
    - `pip install -r requirements.txt`
1. Run `social_media_buzz` module:
    - `python -m social_media_buzz`
1. Check results under `/assets/results/`


# Acknowledgements

Special thanks to François Kawala, Ahlame Douzal, Eric Gaussier, and Eustache Diemert (from Université Joseph Fourier and BestofMedia Group) for providing the data set used here.

I'd also like to thank University of California Irvine for hosting the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php), [where the data set](https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+) can be [downloaded](https://archive.ics.uci.edu/ml/machine-learning-databases/00248/regression.tar.gz). 

# Todo

### Essential

- [x] Load data from file
- [x] Divide data in Training (80%) vs Testing (20%)
- [x] Create linear regression model for a pair of variables (1 predictor)
- [x] Cycle through features
- [x] Get R-squared for each attribute
- [x] Rank attribute based on R-squared value.
- [x] Write short report
 
### Extra

- [x] Create several folds for Training/Testing data (Cross-validation)
- [x] Cycle through folds
- [x] Rank attribute based on testing data accuracy.
- [x] Generate charts
 
### Above and Beyond
- [ ] Fetch data set automatically
- [ ] Compare both rankings automatically
- [ ] Optimize with threads
- [ ] Optimize with Cython?