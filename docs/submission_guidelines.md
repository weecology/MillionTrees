# Submissions
Thank you for submitting to the MillionTrees leaderboards. The format for this benchmark follows the excellent [Wilds Benchmark](https://wilds.stanford.edu/submit/).

We welcome submissions of new algorithms and/or models, and we encourage contributors to test their new methods on as many datasets as applicable. This is valuable even if (or especially if) your method performs well on some datasets but not others.

We also welcome re-implementations of existing methods. On the leaderboards, we distinguish between official submissions (made by the authors of a method) and unofficial submissions (re-implementations by other contributors). Unofficial submissions are equally valuable, especially if the re-implementations achieve better performance than the original implementations because of better tuning or simple tweaks.

All submissions must use the dataset classes and evaluators in the MillionTrees package. In addition, they must report results on 3 random seeds.

Submissions fall into two categories: standard submissions and non-standard submissions.

## Standard submissions

Standard submissions must follow these guidelines:

* Results must be reported on at least 3 random seeds.
* The test set must not be used in any form for model training or selection.
* The validation set must be either the official out-of-distribution (OOD) validation set or, if applicable, the official in-distribution (ID) validation set.
* The validation set should only be used for hyperparameter selection. For example, after hyperparameters have been selected, do not combine the validation set with the training set and retrain the model.
* Training and model selection should not use any additional data, labeled or unlabeled, beyond the official training and validation data.
* To avoid unintended adaptation, models should not use batch statistics during evaluation. BatchNorm is accepted in its default mode (where it uses batch statistics during training, and then fixes them during evaluation).

## Non-standard submissions
Non-standard submissions only need to follow the first two guidelines from above:

* Results must be reported on at least 3 random seeds.
* The test set must not be used in any form for model training or selection.

These submissions will be differentiated from standard submissions in our leaderboards. They are meant for the community to try out different approaches to solving these tasks. Examples of non-standard submissions might include Using unlabeled data from external sources, specialized methods for particular datasets/domains.

### Making a submission

Submitting to the leaderboard consists of two steps: first, uploading your predictions in .csv format, and second, filling up our submission form.

## Submission formatting
Please submit your predictions in .csv format for all datasets except GlobalWheat, and .pth format for the GlobalWheat dataset. The example scripts in the examples/ folder will automatically train models and save their predictions in the right format; see the Get Started page for information on how to use these scripts.

If you are not using the example scripts, see the last section on this page for details on the expected format.

### Step 1: Uploading your predictions
Upload a .tar.gz or .zip file containing your predictions in the format specified above. Feel free to use any standard host for your file (Google Drive, Dropbox, etc.).

Check that your predictions are valid by running the evaluate.py script on them. To do so, run python3 examples/evaluate.py [path_to_predictions] [path_to_output_results] --root_dir [path_to_data].

Please upload a separate .tar.gz or .zip file per method that you are submitting. For example, if you are submitting algorithm A and algorithm B, both of which are evaluated on 6 different datasets, then you should submit two different .tar.gz or .zip files: one corresponding to algorithm A (and containing predictions for all 6 datasets) and the other corresponding to algorithm B (also containing predictions for all 6 datasets.)

### Step 2: Filling out the submission form
Next, fill up the submission form. You will need to fill out one form per .tar.gz/.zip file submitted. The form will ask for the URL to your submission file.

Once these steps have been completed, we will evaluate the predictions using the evaluate.py script and update the leaderboard within a week.