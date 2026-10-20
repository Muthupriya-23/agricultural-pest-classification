
from pathlib import Path
import cv2
import numpy as np
import joblib

from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_DIR = Path(
    r"C:\Users\Priya\Downloads\study-cap&materials\computer_vision\archive"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (128, 128)
MAX_IMAGES_PER_CLASS = 500


def extract_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, IMAGE_SIZE)

    # Module 1: Image preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Module 2: HOG shape and edge features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Module 3: Texture features using LBP
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # Colour features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 8, 8], [0, 180, 0, 256, 0, 256]
    ).flatten()
    color_hist /= (color_hist.sum() + 1e-7)

    return np.concatenate([hog_features, lbp_hist, color_hist])


def main():
    if not DATASET_DIR.exists():
        print("Dataset folder not found:", DATASET_DIR)
        return

    features = []
    labels = []

    class_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        print("No category folders found.")
        return

    print("Reading dataset...")

    for folder in class_folders:
        image_files = [
            path for path in folder.rglob("*")
            if path.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        image_files = image_files[:MAX_IMAGES_PER_CLASS]
        successful = 0

        for image_path in image_files:
            try:
                feature = extract_features(image_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(folder.name)
                    successful += 1
            except Exception as error:
                print("Skipped:", image_path.name, error)

        print(f"{folder.name}: {successful} images processed")

    if len(features) < 2 or len(set(labels)) < 2:
        print("Not enough valid images to train the model.")
        return

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    print("\nTotal images processed:", len(y))
    print("Categories:", sorted(set(y)))

    # Use stratification when every class has enough examples
    _, counts = np.unique(y, return_counts=True)
    stratify_labels = y if counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels
    )

    print("\nTraining model...")

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", class_weight="balanced")
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_DIR / "pest_classifier.joblib")
    joblib.dump(sorted(set(y)), MODEL_DIR / "classes.joblib")

    print("\nModel saved successfully in the models folder.")


if __name__ == "__main__":
    main()
