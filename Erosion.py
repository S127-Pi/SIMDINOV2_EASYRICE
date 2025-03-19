import cv2
import numpy as np



def crop_rice_grains(images, y_keys, output_size=(224, 224), kernel_size=(3,3)):
    """
    Process multiple images stored under a single key in an NPZ file, 
    extract the largest contour (rice grain), and return a list of cropped images.

    Args:
        npz_path (str): Path to the NPZ file.
        output_size (tuple): Size of the output images (default: 224x224).
        kernel_size (tuple): Kernel size for morphological operations.

    Returns:
        list: List of processed images (each as a NumPy array).
    """
    # Load images from the NPZ file
    # images = np.load(npz_path)['kernel_pics'].astype('uint8')
    
    err = False

    # Ensure the images have 4 dimension (B, H, W, C)
    if len(images.shape) == 3:  # images (H, W, C)
        images = np.expand_dims(images, axis=0)  # Convert to (1, H, W, C)
    elif len(images.shape) != 4 or images.shape[-1] not in [1, 3]:  # Unexpected format
        err = True
        print(f"Unexpected image shape: {images.shape}, the source npz file is {y_keys}")
        return images, y_keys, err

    # Store processed images
    processed_images = []


    for i, image in enumerate(images):

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # Create kernel for morphological operations
        kernel = np.ones(kernel_size, np.uint8)

        # Apply opening operation to clean up the mask
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours in the opened image
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find the largest contour (should be the rice grain)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a mask from the largest contour
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

            # Create black background
            result = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)

            # Copy the rice grain onto black background using the mask
            result = cv2.bitwise_and(image, image, mask=mask)

            # Store the processed image
            result = result/255.0 # normalize the image
            processed_images.append(result)
        else:
            #background found
            image = image/255.0 
            # processed_images.append(image)
            # print(f"Skipping image {i}: No rice grain found.")

    # y_keys = [f"{npz_path.split('/')[-1].split('.')[0]}_{i}.{npz_path.split('.')[-1]}" for i in range(len(images))]
    processed_images = np.array(processed_images)
    
    return processed_images, y_keys, err




# def crop_rice_grain(image_path, output_size=(224, 224), kernel_size=(3,3)):
#     # Read the image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     if image is None:
#         raise ValueError("Could not read the image")
        
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply threshold to get binary image
#     _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

#     # Create kernel for morphological operations
#     kernel = np.ones(kernel_size, np.uint8)

#     # Apply opening operation to clean up the mask
#     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,iterations=2)
#     # Find contours in the opened image
#     contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) > 0:
#         # Find the largest contour (should be the rice grain)
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         # Create a mask from the largest contour
#         mask = np.zeros(output_size, dtype=np.uint8)
#         cv2.drawContours(mask, [largest_contour], -1, (255), -1)
        
#         # Create black background
#         result = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
        
#         # Copy the rice grain onto black background using the mask
#         result = cv2.bitwise_and(image, image, mask=mask)
        
#         return result

#     return None

# image_path = "Download/dataset/over/0153_16112024101201_176.png"
# plt.imshow(plt.imread(image_path))
# plt.show()
# # Try with small kernel
# small_kernel_result = crop_rice_grain(image_path, kernel_size=(2,2))

# img_array = small_kernel_result / 255.0  # Rescale to match training preprocessing
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# # Make prediction
# predictions = model.predict(img_array)
# predicted_class_index = np.argmax(predictions[0])
# print(predicted_class_index)

# plt.imshow(small_kernel_result)
# plt.show()