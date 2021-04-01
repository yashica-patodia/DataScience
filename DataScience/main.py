# Imports
from my_package.model import ObjectDetectionModel
from my_package.data import Dataset
from my_package.analysis import plot_boxes
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import matplotlib.pyplot as plt


def experiment(annotation_file, detector, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        detector: The object detector
        transforms: List of transformation classes
        outputs: path of the output folder to store the images
    '''

    # Create the instance of the dataset.
    dataset = Dataset(annotation_file=annotation_file, transforms=None)

    # Iterate over all data items.
    len_dataset = len(dataset)
    for i in range(len_dataset):
        # Get the predictions from the detector.
        print("Predicting the {}th image".format(i))
        image_pred = dataset[i]
        pred_boxes, pred_class, pred_score = detector(image_pred["image"])
        # Draw the boxes on the image and save them.
        image = image_pred["image"]
        output_path = outputs + "all_images/out" + str(i) + ".jpg"
        plot_boxes(image, pred_boxes, pred_class, pred_score, output_path)

    # Do the required analysis experiments.
    Roll_number_last_digit = 7
    print("Analysing the {}th Image".format(Roll_number_last_digit))
    image_pred = dataset[Roll_number_last_digit]
    image = image_pred["image"]
    output_path_analysis = outputs + "image_analysis/output{}.jpg".format(Roll_number_last_digit)
    pred_boxes, pred_class, pred_score = detector(image)
    img = plot_boxes(image, pred_boxes, pred_class, pred_score, output_path_analysis)
    plt.subplot(4, 4, 1)
    #plt.title("Image")
    plt.imshow(img)

    print("Horizontally flipped original image along with the predicted bounding boxes.")
    flip_data = Dataset(annotation_file=annotation_file,transforms=[transforms[0]()])
    flip_dict = flip_data[Roll_number_last_digit]
    pred_boxes, pred_class, pred_score = detector(flip_dict["image"])
    image = flip_dict["image"]
    output_path_flip = outputs + "image_analysis/output_flip{}.jpg".format(Roll_number_last_digit)
    flip_img = plot_boxes(image, pred_boxes, pred_class, pred_score, output_path_flip)
    plt.subplot(4, 4, 2)
    #plt.title("Image")
    plt.imshow(flip_img)

    print("Blurred image (with some degree of blurring) along with the predicted bounding boxes.")
    blur_data = Dataset(annotation_file, [transforms[1](2)])
    blur_dict = blur_data[Roll_number_last_digit]
    pred_boxes, pred_class, pred_score = detector(blur_dict["image"])
    image = blur_dict["image"]
    output_path_blur = outputs + "image_analysis/output_blur{}.jpg".format(Roll_number_last_digit)
    blur_img = plot_boxes(image, pred_boxes, pred_class, pred_score, output_path_blur)
    plt.subplot(4, 4, 3)
    #plt.title("Image")
    plt.imshow(blur_img)
    #
    print("Twice Rescaled image (2X scaled) along with the predicted bounding boxes")
    rescale1_data = Dataset(annotation_file, [transforms[3]((2 * dataset[Roll_number_last_digit]["image"].shape[2],
                                                             2 * dataset[Roll_number_last_digit]["image"].shape[1]))])
    rescale1_dict = rescale1_data[Roll_number_last_digit]
    pred_boxes, pred_class, pred_score = detector(rescale1_dict["image"])
    image = rescale1_dict["image"]
    output_path_rescale1 = outputs + "image_analysis/output_rescale2X{}.jpg".format(Roll_number_last_digit)
    rescale1_img = plot_boxes(image, pred_boxes, pred_class, pred_score, output_path_rescale1)
    plt.subplot(4, 4, 4)
    #plt.title("Image")
    plt.imshow(rescale1_img)

    print("Half Rescaled image (0.5X scaled) along with the predicted bounding boxes")
    rescale2_data = Dataset(annotation_file, [transforms[3]((int(dataset[Roll_number_last_digit]["image"].shape[2] / 2),
                                                             int(dataset[Roll_number_last_digit]["image"].shape[
                                                                     1] / 2)))])
    rescale2_dict = rescale2_data[Roll_number_last_digit]
    pred_boxes, pred_class, pred_score = detector(rescale2_dict["image"])
    image = rescale2_dict["image"]
    output_path_rescale2 = outputs + "image_analysis/output_rescale.5X{}.jpg".format(Roll_number_last_digit)
    rescale2_img = plot_boxes(image, pred_boxes, pred_class, pred_score, output_path_rescale2)
    plt.subplot(4, 4, 5)
    #plt.title("Image")
    plt.imshow(rescale2_img)
    #
    print("90 degree right rotated image along with the predicted bounding boxes")
    rotate1_data = Dataset(annotation_file, [transforms[2](90)])
    rotate1_dict = rotate1_data[Roll_number_last_digit]
    pred_boxes, pred_class, pred_score = detector(rotate1_dict["image"])
    image =rotate1_dict["image"]
    output_path_rotate = outputs + "image_analysis/output_rotate90{}.jpg".format(Roll_number_last_digit)
    rotate1_img = plot_boxes(image, pred_boxes, pred_class, pred_score, output_path_rotate)
    plt.subplot(4, 4, 6)
    #plt.title("Image")
    plt.imshow(rotate1_img)

    print("45 degree right rotated image along with the predicted bounding boxes")
    rotate1_data = Dataset(annotation_file, [transforms[2](45)])
    rotate1_dict = rotate1_data[Roll_number_last_digit]
    pred_boxes, pred_class, pred_score = detector(rotate1_dict["image"])
    image = rotate1_dict["image"]
    output_path_rotate = outputs + "image_analysis/output_rotate45{}.jpg".format(Roll_number_last_digit)
    rotate1_img = plot_boxes(image, pred_boxes, pred_class, pred_score, output_path_rotate)
    plt.subplot(4, 4, 7)
    #plt.title("Image")
    plt.imshow(rotate1_img)


    save_path_file = outputs + 'image_analysis/analysis.png'
    plt.savefig(save_path_file)
    plt.show()


def main():
    detector = ObjectDetectionModel()
    experiment('/content/drive/MyDrive/NewDS/data/annotations.jsonl', detector, [FlipImage, BlurImage, RotateImage, RescaleImage, CropImage],
               '/content/drive/MyDrive/NewDS/outputs/')  # Sample arguments to call experiment()


if __name__ == '__main__':
    main()

# learn how to structure the code
# where should the setup.py be and __init__.py be?
