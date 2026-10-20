import kagglehub

# Download latest version
path = kagglehub.dataset_download("roua91/isic-melanoma-dataset-resized-to-224x224")

print("Path to dataset files:", path)