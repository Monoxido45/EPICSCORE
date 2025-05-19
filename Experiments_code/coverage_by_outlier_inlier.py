import pickle
import numpy as np
import pandas as pd
import torch
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split
import gc

# MDN method and APS score and method
from Epistemic_CP.epistemic_cp import ECP_split, APSSplit
from Epistemic_CP.scores import APSScore

# for anomaly detection
from sklearn.neighbors import LocalOutlierFactor

# feature extraction libraries
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# base models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
from tqdm import tqdm

import tarfile
import urllib.request

##################################################################
# Enable LaTeX rendering in matplotlib
plt.rcParams["text.usetex"] = True


# functions to make feature extraction and pre-processing
def feature_extraction(
    images,
    model,
    device,
    batch_size=500,
    preprocessing_batch_size=20,
):
    # Send the model to the GPU
    model = model.to(device)
    # function to extract features
    feature_extractor = create_feature_extractor(
        model, return_nodes={"avgpool": "features"}
    )
    # preprocessing functions
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Transform the dataset
    def transform_image(image):
        image = preprocess(image)
        return image

    # feature list
    feature_list = []

    for i in tqdm(
        range(0, len(images), batch_size), desc="Performing feature extraction"
    ):
        batch = images[i : i + batch_size]
        transformed_batches = []

        for i in range(0, len(batch), preprocessing_batch_size):
            pre_batch = batch[i : i + preprocessing_batch_size]
            transformed_batch = torch.stack(
                [transform_image(image) for image in pre_batch]
            )
            transformed_batches.append(transformed_batch)

        # Concatenate all batches into a single tensor
        transformed_data = torch.cat(transformed_batches)

        # delete original batch for memory purposes
        del batch
        gc.collect()

        # Move the model to the same device as the data
        model = model.to(device)

        # Move the transformed_train_data to the same device as the model
        transformed_data = transformed_data.to(device)

        # Extract features
        with torch.no_grad():
            features = feature_extractor(transformed_data)["features"]

        # deleting transformed_data
        transformed_data = transformed_data.cpu()
        # deleting object
        del transformed_data
        gc.collect()

        # Flatten the features to 2D
        features_2d = features.view(features.size(0), -1)
        features_numpy = features_2d.cpu().numpy()

        # concatenating
        feature_list.append(features_numpy)
    return feature_list


def ssc_metric(
    pred_sets,
    label_true,
    label_numbers,
    G=20,
    violation=False,
    alpha=0.1,
):
    # dividing into G subgroups
    card_idx = np.arange(0, G)
    i = 0
    prob_array = np.zeros(G)
    # computing each prediction set cardinality
    card_pred = pred_sets.sum(axis=1)
    for card in card_idx:
        if i + 1 == G:
            filter = np.where(card_pred > (card + 1))[0]
        else:
            filter = np.where(card_pred == (card + 1))[0]

        bin_size = filter.shape[0]

        if bin_size > 0:
            # filtering the prediction sets
            pred_filter = pred_sets[filter, :]
            coverage_array = np.zeros(bin_size)
            label_filter = label_true[filter]

            for j in range(bin_size):
                idxs = label_numbers[pred_filter[j, :]]
                coverage_array[j] = int(np.isin(label_filter[j], idxs))

            prob_array[i] = np.mean(coverage_array)
        else:
            prob_array[i] = np.nan
        i += 1

    if violation:
        # filtering arrays
        prob_array = prob_array[~np.isnan(prob_array)]
        sscv = np.max(np.abs(prob_array - (1 - alpha)))
        return sscv
    else:
        # selecting minimum prob
        prob_min = np.nanmin(prob_array)
        return prob_min


##################################################################
# Pre-processing data and feature extraction
# importing resnet model
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

# Send the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# path
original_path = os.getcwd()


# URL for the CIFAR-100 dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
filename = "cifar-100-python.tar.gz"
folder_path = "cifar-100-python"

# Download the dataset
urllib.request.urlretrieve(url, filename)

with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()
os.remove(filename)


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


# Extracting the dataset
folder_path = "cifar-100-python"

files = os.listdir(folder_path)

data_list = []
labels_list = []
for file_name in files:
    if file_name != "file.txt~":
        metadata = unpickle(folder_path + "/" + file_name)
        if file_name == "meta":
            label_names = metadata[b"fine_label_names"]
        else:
            data = metadata[b"data"]
            labels = metadata[b"fine_labels"]
            data_list.append(data)
            labels_list.append(labels)

# decoding labels
label_names = [label.decode("utf-8") for label in label_names]
print(label_names)

# concatenating data
data_all = np.concatenate(data_list, axis=0)
labels_all = np.concatenate(labels_list, axis=0)

# reshaping data so it fits inside a 32x32 matrix
n_obs = data_all.shape[0]
data_recoded = data_all.reshape(n_obs, 3, 32, 32).transpose(0, 2, 3, 1)


# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    data_recoded, labels_all, test_size=0.1, random_state=45
)

# Further split the training data into training and calibration sets
train_images, cal_images, train_labels, cal_labels = train_test_split(
    train_images, train_labels, test_size=0.5, random_state=45
)

# performing feature extraction
# for training set
train_list = feature_extraction(train_images, model, device)
X_train = np.concatenate(train_list, axis=0)
del train_list
gc.collect()

# for calibration set
cal_list = feature_extraction(cal_images, model, device)
X_cal = np.concatenate(cal_list, axis=0)
del cal_list
gc.collect()

# for testing set
test_list = feature_extraction(test_images, model, device)
X_test = np.concatenate(test_list, axis=0)
del test_list
gc.collect()

############################################################
# Fitting base model and conformal approaches
# Random forest
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

# Fit the model on the training data
rf_model.fit(X_train, train_labels)


np.random.seed(45)
torch.cuda.manual_seed(35)
torch.manual_seed(35)

# fitting APS
alpha = 0.2
aps_obj = APSSplit(rf_model, alpha=alpha, is_fitted=True)
aps_obj.fit(X_train, train_labels)
aps_obj.calibrate(X_cal, cal_labels)

# fitting EPICSCORE MDN
ecp_obj = ECP_split(
    APSScore,
    base_model=rf_model,
    alpha=alpha,
    is_fitted=True,
)

ecp_obj.fit(X_train, train_labels)
ecp_obj.calib(
    X_cal,
    cal_labels,
    num_components=3,
    dropout_rate=0.5,
    epistemic_model="MC_dropout",
    hidden_layers=[64, 64, 32],
    patience=50,
    epochs=2000,
    scale=True,
    batch_size=135,
    verbose=2,
    normalize_y=True,
    type="gaussian",
)

# obtaining prediction sets
# label strings
label_strings = np.array(label_names)

# Predicting all testing samples
pred_sets_aps = aps_obj.predict(X_test)
pred_sets_mdn = ecp_obj.predict(X_test)

#################################################################
# Evaluating the prediction sets by outlier and inlier
# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne_test = tsne.fit_transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_tsne_test)

# Use Local Outlier Factor for anomaly detection on scaled data
lof = LocalOutlierFactor(n_neighbors=25, contamination=0.1)
y_pred = lof.fit_predict(X_test_scaled)

# outliers indexes
outlier_images = test_images[y_pred == -1]
outlier_indexes = np.where(y_pred == -1)[0]

# outlier labels
most_cont_labels = test_labels[outlier_indexes]

# Identify also the inliers (non-outliers)
inlier_indexes = np.setdiff1d(np.arange(len(test_images)), outlier_indexes)

# selecting contamination scores
contamination_scores = lof.negative_outlier_factor_[outlier_indexes]
outlier_mdn = pred_sets_mdn[outlier_indexes]
outlier_aps = pred_sets_aps[outlier_indexes]
most_cont_labels = test_labels[outlier_indexes]

# Select the top 50% of outliers based on contamination scores
size_outliers = 150
top_outlier_indexes = np.argsort(contamination_scores)[:size_outliers]
outlier_mdn_sets = outlier_mdn[top_outlier_indexes]
outlier_aps_sets = outlier_aps[top_outlier_indexes]
outlier_labels = most_cont_labels[top_outlier_indexes]


# Select the 4 most common inliers based on the highest scores (least negative)
inlier_scores = lof.negative_outlier_factor_[inlier_indexes]
inlier_mdn = pred_sets_mdn[inlier_indexes]
inlier_aps = pred_sets_aps[inlier_indexes]
most_common_labels = test_labels[inlier_indexes]

# returning the top 150 of most inliers observations
size = 150
most_inlier_idxs = np.argsort(inlier_scores)[::-1][:size]
inlier_mdn_sets = inlier_mdn[most_inlier_idxs]
inlier_aps_sets = inlier_aps[most_inlier_idxs]
inlier_labels = most_common_labels[most_inlier_idxs]

# computing coverage for inlier and outlier separately
labels = np.arange(0, 100)
# first for inlier
coverage_inlier_mdn = 0
coverage_inlier_aps = 0
for i in range(inlier_labels.shape[0]):
    label = inlier_labels[i]
    pred_set_mdn = labels[inlier_mdn_sets[i]]
    pred_set_aps = labels[inlier_aps_sets[i]]

    is_in_mdn = label in pred_set_mdn
    is_in_aps = label in pred_set_aps

    coverage_inlier_mdn += is_in_mdn
    coverage_inlier_aps += is_in_aps

coverage_inlier_mdn /= inlier_labels.shape[0]
coverage_inlier_aps /= inlier_labels.shape[0]

print("Mean Coverage for Inliers (MDN):", coverage_inlier_mdn)
print("Mean Coverage for Inliers (APS):", coverage_inlier_aps)

# next for outlier
coverage_outlier_mdn = 0
coverage_outlier_aps = 0
for i in range(outlier_labels.shape[0]):
    label = outlier_labels[i]
    pred_set_mdn = labels[outlier_mdn_sets[i]]
    pred_set_aps = labels[outlier_aps_sets[i]]

    is_in_mdn = label in pred_set_mdn
    is_in_aps = label in pred_set_aps

    coverage_outlier_mdn += is_in_mdn
    coverage_outlier_aps += is_in_aps

coverage_outlier_mdn /= outlier_labels.shape[0]
coverage_outlier_aps /= outlier_labels.shape[0]

print("Mean Coverage for Outliers (MDN):", coverage_outlier_mdn)
print("Mean Coverage for Outliers (APS):", coverage_outlier_aps)


# Compute mean coverage for inlier sets
mean_coverage_inlier_mdn = np.mean(
    [
        label in label_strings[inlier_mdn_sets[i]]
        for i, label in enumerate(inlier_labels)
    ]
)
mean_coverage_inlier_aps = np.mean(
    [
        label in label_strings[inlier_aps_sets[i]]
        for i, label in enumerate(inlier_labels)
    ]
)

# Compute mean coverage for outlier sets
mean_coverage_outlier_mdn = np.mean(
    [
        label in label_strings[outlier_mdn_sets[i]]
        for i, label in enumerate(outlier_labels)
    ]
)
mean_coverage_outlier_aps = np.mean(
    [
        label in label_strings[outlier_aps_sets[i]]
        for i, label in enumerate(outlier_labels)
    ]
)

print("Mean Coverage for Inliers (MDN):", mean_coverage_inlier_mdn)
print("Mean Coverage for Inliers (APS):", mean_coverage_inlier_aps)
print("Mean Coverage for Outliers (MDN):", mean_coverage_outlier_mdn)
print("Mean Coverage for Outliers (APS):", mean_coverage_outlier_aps)

# Plotting the coverages
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Inlier coverage subplot
axes[0].bar(
    ["MDN", "APS"],
    [coverage_inlier_mdn, coverage_inlier_aps],
    color=["darkred", "green"],
)
axes[0].axhline(y=0.8, color="black", linestyle="--", label="Nominal level")
axes[0].set_ylim(0.5, 0.9)
for bar in axes[0].patches:
    bar.set_alpha(0.7)
axes[0].set_title("Inlier Coverage")
axes[0].set_ylabel("Coverage")
axes[0].legend()

# Outlier coverage subplot
axes[1].bar(
    ["MDN", "APS"],
    [coverage_outlier_mdn, coverage_outlier_aps],
    color=["darkred", "green"],
)
axes[1].axhline(y=0.8, color="black", linestyle="--", label="Nominal level")
axes[1].set_title("Outlier Coverage")
axes[1].legend()
axes[1].set_ylim(0.6, 0.85)
for bar in axes[1].patches:
    bar.set_alpha(0.7)
# Remove the nominal level legend from individual subplots
axes[0].legend().remove()
axes[1].legend().remove()

# Add a single legend for the nominal level line
fig.legend(["Nominal level"], loc="upper center", ncol=1, frameon=False)

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# Plotting the coverages
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Inlier coverage subplot
axes[0].bar(
    ["MDN", "APS"],
    [np.abs(coverage_inlier_mdn - 0.8), np.abs(coverage_inlier_aps - 0.8)],
    color=["darkred", "green"],
)

axes[0].set_ylim(0, 0.15)
for bar in axes[0].patches:
    bar.set_alpha(0.65)
axes[0].set_title("Inliers")
axes[0].set_ylabel("Coverage deviation")

# Outlier coverage subplot
axes[1].bar(
    ["MDN", "APS"],
    [np.abs(coverage_outlier_mdn - 0.8), np.abs(coverage_outlier_aps - 0.8)],
    color=["darkred", "green"],
)
axes[1].set_title("Outlier")
axes[1].set_ylim(0, 0.1)
for bar in axes[1].patches:
    bar.set_alpha(0.65)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
