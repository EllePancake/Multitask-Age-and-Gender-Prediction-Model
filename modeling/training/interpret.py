import torch
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from utils_data_loader import get_custom_data_loaders_v1, get_custom_data_loaders_v2
from utils_data_loader import get_custom_data_loaders_v3, get_custom_data_loaders_v4
from base_model import MultiTaskResNet
from evaluator import load_model_from_checkpoint


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, images):
        """
        Converts the LIME input (numpy array) to a tensor and performs model
        inference.

        Args:
            images (np.ndarray): Batch of input images in numpy format (N, H,
            W, C).

        Returns:
            np.ndarray: Predicted gender probabilities or age predictions for
            each image.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        images_torch = torch.stack([transform(image) for image in images],
                                   dim=0)

        device = next(self.model.parameters()).device
        images_torch = images_torch.to(device)

        with torch.no_grad():
            age_preds, gender_preds = self.model(images_torch)

        gender_probs = torch.softmax(gender_preds, dim=1).cpu().numpy()

        age_preds_np = age_preds.squeeze().cpu().numpy()

        return np.hstack([gender_probs, age_preds_np[:, np.newaxis]])

    def predict_for_feature(self, images, feature_name):
        """
        Predict either gender or age based on the feature name.

        Args:
            images (np.ndarray): Batch of input images in numpy format (N, H,
            W, C).
            feature_name (str): Feature to interpret ('gender' or 'age').

        Returns:
            np.ndarray: Predicted feature values for each image.
        """
        images_torch = torch.stack(
            [transforms.ToTensor()(image) for image in images], dim=0)
        images_torch = images_torch.to(next(self.model.parameters()).device)

        with torch.no_grad():
            age_preds, gender_preds = self.model(images_torch)

        if feature_name == 'gender':
            return torch.softmax(gender_preds, dim=1).cpu().numpy()
        elif feature_name == 'age':
            return age_preds.squeeze().cpu().numpy()
        else:
            raise ValueError("feature_name must be either 'gender' or 'age'")


def interpret_sample(model_wrapper, image, label, feature_name='gender'):
    """
    Interpret the model's prediction using LIME for a given sample.

    Args:
        model_wrapper (ModelWrapper): Wrapper object around the trained model
        for LIME.
        image (np.ndarray): The input image in numpy format.
        label (int): The true label for the image.
        feature_name (str): Feature to interpret ('gender' or 'age').
    """
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=image,
        classifier_fn=model_wrapper.predict,
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )

    if feature_name == 'gender':
        feature_label = explanation.top_labels[0]
    else:
        feature_label = explanation.top_labels[1]

    temp, mask = explanation.get_image_and_mask(
        label=feature_label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title(f'LIME Interpretation for {feature_name}')
    plt.imshow(mark_boundaries(temp / 255.0, mask))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interpret model predictions using LIME.")
    parser.add_argument('--version', type=str, required=True,
                        help="Version of the model to interpret)")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help="Number of samples to interpret.")

    args = parser.parse_args()
    version = args.version
    checkpoint_path = args.checkpoint_path
    num_samples = args.num_samples

    if version == 'v1':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v1()
    elif version == 'v2':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v2()
    elif version == 'v3':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v3()
    elif version == 'v4':
        model_class = MultiTaskResNet
        train_loader, val_loader, test_loader = get_custom_data_loaders_v4()
    else:
        raise ValueError(f"Unsupported version: {version}")

    model = load_model_from_checkpoint(model_class, checkpoint_path)
    model_wrapper = ModelWrapper(model)

    data_iter = iter(test_loader)
    for i in range(num_samples):
        images, labels = next(data_iter)
        image_np = images[0].permute(1, 2, 0).numpy() * 255
        label = labels[0]

        print(
            f"Interpreting sample {i + 1}/{num_samples} for gender prediction")
        interpret_sample(model_wrapper, image_np.astype(np.uint8), label,
                         feature_name='gender')

        print(f"Interpreting sample {i + 1}/{num_samples} for age prediction.")
        interpret_sample(model_wrapper, image_np.astype(np.uint8), label,
                         feature_name='age')
