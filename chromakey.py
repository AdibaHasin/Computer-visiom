import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load and resize an image
def load_and_resize_image(filename):
    image = cv2.imread(filename)
    max_dim = max(image.shape[:2])
    scale = min(1280 / image.shape[1], 720 / image.shape[0])
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    return image
def get_channel_names(color_space):
    if color_space=="-XYZ":
        return "X", "Y", "Z"
    elif color_space=="-Lab":
        return "L", "a", "b"
    elif color_space=="-HSB":
        return "H", "S", "B"
    else:
        return "", "", ""

def main():
    if len(sys.argv)!=3:
       print("Usage: for task 1- python Chromakey –XYZ|-Lab|-YCrCb|-HSB imagefile, for task 2- Chromakey scenicImageFile greenScreenImagefile")
       return

    if len(sys.argv) == 3 and (sys.argv[1] == "-XYZ" or sys.argv[1] == "-Lab" or sys.argv[1] == "-YCrCb" or sys.argv[1] == "-HSB"):
        image_filename = sys.argv[2]
        color_space_option = sys.argv[1]

        image = load_and_resize_image(image_filename)
        channel_names = get_channel_names(color_space_option)
        
        if color_space_option == "-XYZ":
            color_space_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        elif color_space_option == "-Lab":
            color_space_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        elif color_space_option == "-YCrCb":
            color_space_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        elif color_space_option == "-HSB":
            color_space_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            print("Invalid color space option.")
            return  # Exit the program

        ch1, ch2, ch3 = cv2.split(color_space_image)
        ch1_grey = ch1
        ch2_grey = ch2
        ch3_grey = ch3

        max_width = 1280
        max_height = 720
        height, width, _ = image.shape
        scale_factor = min(max_width / (2 * width), max_height / (2 * height))
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the original image and channel images
        image_resized = cv2.resize(image, (new_width, new_height))
        ch1_resized = cv2.resize(cv2.cvtColor(ch1_grey, cv2.COLOR_GRAY2BGR), (new_width, new_height))
        ch2_resized = cv2.resize(cv2.cvtColor(ch2_grey, cv2.COLOR_GRAY2BGR), (new_width, new_height))
        ch3_resized = cv2.resize(cv2.cvtColor(ch3_grey, cv2.COLOR_GRAY2BGR), (new_width, new_height))

        # Add labels
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(image_resized, 'Original image', (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(ch1_resized, channel_names[0], (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(ch2_resized, channel_names[1], (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(ch3_resized, channel_names[2], (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Stack the resized images horizontally
        combined_top = np.hstack((image_resized, ch1_resized))
        combined_bottom = np.hstack((ch2_resized, ch3_resized))

        # Resize the top part to match the width of the bottom part
        combined_top_resized = cv2.resize(combined_top, (combined_bottom.shape[1], combined_top.shape[0]))

        # Stack the top and bottom parts vertically
        combined = np.vstack((combined_top_resized, combined_bottom))

        cv2.imshow("Combined", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       

    elif len(sys.argv) == 3:
        # Task Two
        scenic_image_path = sys.argv[1]
        green_screen_image_path = sys.argv[2]
        green_screen_photo = load_and_resize_image(green_screen_image_path)
        scenic_photo = load_and_resize_image(scenic_image_path)

        # Calculate the aspect ratio of the scenic photo
        scenic_aspect_ratio = scenic_photo.shape[1] / scenic_photo.shape[0]

        # Calculate the new width and height for resizing while maintaining aspect ratio
        max_width = 1280
        max_height = 720
        height, width, _ = scenic_photo.shape
        scale_factor = min(max_width / (2 * width), max_height / (2 * height))
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

       

        # Define green color range (in HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Convert green screen photo to HSV
        green_screen_hsv = cv2.cvtColor(green_screen_photo, cv2.COLOR_BGR2HSV)

        # Create a mask for green pixels
        green_mask = cv2.inRange(green_screen_hsv, lower_green, upper_green)

        # Invert the mask
        inverse_mask = cv2.bitwise_not(green_mask)

        # Extract person from green screen photo
        person_extracted = cv2.bitwise_and(green_screen_photo, green_screen_photo, mask=inverse_mask)

        # Create a white background canvas
        white_background = np.ones_like(green_screen_photo) * 255

        img = green_screen_photo.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a_channel = lab[:,:,1]
        th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        masked = cv2.bitwise_and(img, img, mask = th)    # contains dark background
        m1 = masked.copy()
        m1[th==0]=(255,255,255) 
        mlab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
        dst = cv2.normalize(mlab[:,:,1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Set the person on the white background
        #person_on_white = white_background.copy()
        #person_on_white[inverse_mask > 0] = person_extracted[inverse_mask > 0]
        img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
        img[th==0]=(255,255,255)
        threshold_value = 100
        dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
        mlab2 = mlab.copy()
        mlab[:,:,1][dst_th == 255] = 127
        img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
        img[th==0]=(255,255,255)

        # Define the lower and upper green color thresholds for the second code snippet
        lower_green_second = np.array([35, 50, 50])
        upper_green_second = np.array([85, 255, 255])

        # Convert green screen image to HSV color space for the second code snippet
        hsv_image_second = cv2.cvtColor(green_screen_photo, cv2.COLOR_BGR2HSV)

        # Create a mask for green color range for the second code snippet

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a mask
        _, mask = cv2.threshold(gray_image, 254.99999999999998, 256, cv2.THRESH_BINARY)

        # Perform morphological operations (optional)

        # Invert the mask
        inverted_mask = cv2.bitwise_not(mask)

        # Extract the person from the original image using the inverted mask
        person_extracted = cv2.bitwise_and(img, img, mask=inverted_mask)
        green_mask_second = cv2.inRange(hsv_image_second, lower_green_second, upper_green_second)

        # Invert the mask to get non-green areas for the second code snippet
        # non_green_mask_second = cv2.bitwise_not(green_mask_second)

        # Extract the person from green screen image for the second code snippet
        # person_second = cv2.bitwise_and(img, img, mask=non_green_mask_second)

        # Calculate the aspect ratio of the person for the second code snippet
        person_height, person_width, _ = person_extracted.shape
        person_aspect_ratio = person_width / person_height

        # Resize the scenic photo to match the size of the person
        scenic_resized = cv2.resize(scenic_photo, (person_width, person_height))

        # Create a mask for the person area for the second code snippet
        person_mask = cv2.cvtColor(person_extracted, cv2.COLOR_BGR2GRAY)
        ret, person_mask = cv2.threshold(person_mask, 1, 255, cv2.THRESH_BINARY)

        # Invert the person mask for the second code snippet
        person_mask_inv = cv2.bitwise_not(inverted_mask)

        # Extract the background area from the resized scenic photo for the second code snippet
        background_area = cv2.bitwise_and(scenic_resized, scenic_resized, mask=person_mask_inv)

        # Combine the person and background images for the second code snippet
        max_width = 1280
        max_height = 720
        height, width, _ = scenic_photo.shape
        scale_factor = min(max_width / (2 * width), max_height / (2 * height))
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        final_image = cv2.add(person_extracted, background_area)
        green_screen_photo_resized = cv2.resize(green_screen_photo, (new_width, new_height))
        img_resized = cv2.resize(img, (new_width, new_height))
        scenic_resized_2 = cv2.resize(scenic_resized, (new_width, new_height))
        final_resized = cv2.resize(final_image, (new_width, new_height))

        # Combine the images for displaying without gaps
        combined_top = np.hstack((green_screen_photo_resized, img_resized))
        combined_bottom = np.hstack((scenic_resized_2, final_resized))

        combined_top_resized = cv2.resize(combined_top, (combined_bottom.shape[1], combined_top.shape[0]))
        combined_image = np.vstack((combined_top_resized, combined_bottom))

        # Display the combined image using matplotlib
       
        cv2.imshow("Combined Image", combined_image)
       
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid arguments. Usage:")
        print("For Task One: Chromakey.py imagefile -XYZ|-Lab|-YCrCb|-HSB")
        print("For Task Two: Chromakey.py greenScreenPhotoFile scenicPhotoFile")

if __name__ == "__main__":
    main()
