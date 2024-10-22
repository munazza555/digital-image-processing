import cv2
import numpy as np
import random

# Class definition for Image Category
class ImageCategory:
    def __init__(self, category_name, image_paths):
        self.category_name = category_name
        self.image_paths = image_paths

    # Function to load and convert an image to grayscale
    def load_image(self, image_path):
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            return None, None  # Return None if image loading fails
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray_image

    # Function to calculate texture using Laplacian of Gaussian (LoG)
    def calculate_texture(self, image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        texture = np.var(laplacian)  # Calculate the variance as a measure of texture
        return texture

    # Function to randomly select an image from the category
    def get_random_image(self):
        random_image_path = random.choice(self.image_paths)
        return random_image_path

    # Function to compare two textures
    def compare_textures(self, texture1, texture2, threshold=0.1):
        return abs(texture1 - texture2) < threshold

    # Function to compute Euclidean distance between two images
    def calculate_euclidean_distance(self, image1, image2):
        # Flatten both images into 1D arrays
        image1_flat = image1.flatten()
        image2_flat = image2.flatten()
        # Calculate the Euclidean distance
        distance = np.linalg.norm(image1_flat - image2_flat)
        return distance


# Defining image paths for each category
cat_images = [
    "cat1.jfif", "cat2.jfif", "cat3.jfif", "cat4.jfif", "cat5.jfif"
]

dog_images = [
    "dog1.jfif", "dog2.jfif", "dog3.jfif", "dog4.jfif", "dog5.jfif"
]

lion_images = [
    "lion1.jfif", "lion2.jfif", "lion3.jfif", "lion4.jfif", "lion5.jfif"
]

# Create instances of ImageCategory class for each category
cat_category = ImageCategory("Cat", cat_images)
dog_category = ImageCategory("Dog", dog_images)
lion_category = ImageCategory("Lion", lion_images)

# Randomly select an image from a known category (let's use cat as the reference image)
reference_image_path = cat_category.get_random_image()
reference_image, gray_reference_image = cat_category.load_image(reference_image_path)

# Check if reference image was loaded properly
if reference_image is None:
    print("Reference image could not be loaded. Exiting...")
else:
    # Calculate texture of the reference image
    reference_texture = cat_category.calculate_texture(gray_reference_image)

    # Randomly select an image from any of the three categories (cat, dog, lion)
    all_categories = [cat_category, dog_category, lion_category]
    random_category = random.choice(all_categories)
    random_image_path = random_category.get_random_image()
    random_image, gray_random_image = random_category.load_image(random_image_path)

    # Check if random image was loaded properly
    if random_image is None:
        print("Random image could not be loaded. Exiting...")
    else:
        # Calculate texture of the randomly selected image
        random_texture = random_category.calculate_texture(gray_random_image)

        # Compare the textures of the reference image and the random image
        if random_category.compare_textures(reference_texture, random_texture):
            print(f"Texture match found! The random image belongs to the {random_category.category_name} category.")
        else:
            print(f"Textures do not match! The random image belongs to the {random_category.category_name} category.")

        # Calculate and display the Euclidean distance between the two images
        euclidean_distance = random_category.calculate_euclidean_distance(gray_reference_image, gray_random_image)
        print(f"Euclidean distance between reference image and random image: {euclidean_distance}")

        # Display both images (reference and random)
        cv2.imshow(f"Reference Image (Cat)", reference_image)
        cv2.imshow(f"Random Image ({random_category.category_name})", random_image)

        # Wait for a key press and close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
