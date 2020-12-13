# EECS 504, Fall 2020, University of Michigan
# Team: The Flatteners

# For using this function, please add the following optional option --mathWorksheet with a path to an image if you want to use a different sheet than the default.

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser()
# For parsing commandline arguments
parser.add_argument("--mathWorksheet", type=str, default='./MathWorksheets/0002.png', help='the math worksheet file')
args = parser.parse_args()

# Perform math operation on two values of a math problem
def fcn_math_operation(math_str, values):
    if math_str == 'plus':
        output = values[0] + values[1]
    elif math_str == 'minus':
        output = values[0] - values[1]
    elif math_str == 'times':
        output = values[0] * values[1]
    else:
        eps = 0.001
        output = values[0] / max(values[1], eps)
    
    return output

def generate_answer_key(img_rgb):
    print('Generating Answer Key!!')

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    clean_output_img_rgb = img_rgb.copy()

    template_files = glob.glob('./AnswerKey/num_*.png')
    

    values = [] # values is a list of dictionary items of the detections

    num_digits_list = ['./AnswerKey/num_'+str(i)+'.png' for i in range(10)]
    numb_types = [str(i) for i in range(10)]
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    cluster_mask = np.zeros(img_rgb.shape[:2], np.uint8)
    for i in template_files:
        
        template = cv2.imread(i,0)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

        # Calibrations of threshold for the template matching algorithm
        # Some digits/symbols require special tuning due to shape similarity
        threshold = 0.8
        if i == './AnswerKey/num_plus.png' or i == './AnswerKey/num_divide.png':
            threshold = 0.93
        elif i == './AnswerKey/num_minus.png':
            threshold = 0.9
        elif i == './AnswerKey/num_times.png':
            threshold = 0.92
        elif i == './AnswerKey/num_0.png' or i == './AnswerKey/num_1.png' or i == './AnswerKey/num_9.png':
            threshold = 0.85
        elif i == './AnswerKey/num_3.png' or i == './AnswerKey/num_8.png':
            threshold = 0.88
        elif i == './AnswerKey/um_5.png':
            threshold = 0.92
        elif i == './AnswerKey/num_6.png':
            threshold = 0.94

        # Location of the template within the worksheet
        loc = np.where( res >= threshold)

        # x,y points of those locations
        for pt in zip(*loc[::-1]):
            if mask[pt[1] + int(round(h/2)), pt[0] + int(round(w/2))] != 255:
                mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
                # If digit or symbol not already detected, apply white to a mask to show it is detected and to prevent any further duplicate detections
                # For visuals, defining unique color bounding boxes to show algorithm is detecting the digits and symbols
                # Saving the type of digit (its number) or symbol and location into a dictionary item that then will be converted to a Pandas DataFrame
                if i == './AnswerKey/num_plus.png':
                    color_show = (255,0,0) # blue
                    values.append({'type' : 'plus', 'x' : pt[0], 'y' : pt[1]})
                elif i == './AnswerKey/num_divide.png':
                    color_show = (0,255,0) # green
                    values.append({'type' : 'divide', 'x' : pt[0], 'y' : pt[1]})
                elif i == './AnswerKey/num_minus.png':
                    color_show = (211,0,148) # purple
                    values.append({'type' : 'minus', 'x' : pt[0], 'y' : pt[1]})
                elif i == './AnswerKey/num_times.png':
                    color_show = (0,204,204) # yellow
                    values.append({'type' : 'times', 'x' : pt[0], 'y' : pt[1]})
                else:
                    color_show = (0,0,255) # red
                    curr_digit = num_digits_list.index(i)
                    values.append({'type' : str(curr_digit), 'x' : pt[0], 'y' : pt[1]})

                # Draw a rectangle around the matched digits and symbols
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), color_show, 1)

    # Write the mask image
    #cv2.imwrite(args.mathWorksheet[:-4] + '_mask_detections.png',mask)

    # Answer's font
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img_rgb, 'ANSWER KEY', (675,100), font, 0.75, (0,0,255), 3, cv2.LINE_AA)
    cv2.putText(clean_output_img_rgb, 'ANSWER KEY', (675,100), font, 0.75, (0,0,255), 3, cv2.LINE_AA)

    df = pd.DataFrame(values)

    X = df[['x', 'y']].to_numpy()

    num_problems = 60
    # Centroid positions initializations
    #Xinit = np.array([[145*j+238, 80*i] for i in range(1,11) for j in range(0, 6)]).astype(np.float64)
    # Utilize KMeans algorithm to cluster the detections into individual math problems on the worksheet page
    kmeans = KMeans(n_clusters=num_problems)#, init=Xinit)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    # plot the cluster assignments and cluster centers

    cluster_centroids = kmeans.cluster_centers_

    # Plot to see the different detections as clusters
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="hot")
    h_cluster_mask = 35
    w_cluster_mask = 35
    overlap_mask = 0
    for i in cluster_centroids:
        if cluster_mask[int(i[1] + int(round(h_cluster_mask/2))), int(i[0] + int(round(w_cluster_mask/2)))] != 255:
            # If the cluster rectangle is not white, show white mask to represent as cluster already detected

            cluster_mask[int(i[1]):int(i[1]+h_cluster_mask), int(i[0]):int(i[0]+w_cluster_mask)] = 255
            top_left_pt = (int(i[0] - 35), int(i[1] - 35))
            bottom_right_pt = (int(i[0] + 35), int(i[1] + 35))
            color_show = (0,0,0)
            cv2.rectangle(img_rgb, top_left_pt, bottom_right_pt, color_show, 3)
        else:
            overlap_mask = 1


    for i in np.unique(kmeans.labels_):
        problem_df = df[['type', 'x', 'y']][kmeans.labels_ == i]
        condition = (problem_df.type == 'plus') | (problem_df.type == 'minus') | (problem_df.type == 'times') | (problem_df.type == 'divide')
        # Remove all math operations and only look at the numbers
        numbers_only = problem_df[~condition]
        numbers_only_sorted = numbers_only.sort_values(by = ['y', 'x'])
        # Get the desired math operation
        math_operation = problem_df[condition]
        num_positions = numbers_only_sorted[['x', 'y']].to_numpy()
        # If there is incorrect detection and two operations are in one cluster or the cluster masks overlap, display ERROR on the output worksheet
        if len(math_operation.type.tolist()) > 1 or overlap_mask == 1:
            cv2.rectangle(img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
            cv2.rectangle(clean_output_img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
            textsize = cv2.getTextSize('Error!!', font, 1, 2)[0]
            textX = (img_rgb.shape[1] - textsize[0]) / 2
            textY = (img_rgb.shape[0] + textsize[1]) / 2
            cv2.putText(img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
            cv2.putText(clean_output_img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
            break

        if len(num_positions) == 2: # math operation on single digits

            # Get the single digit numbers that exist in the problem
            problem_vals = numbers_only_sorted[['type']].to_numpy()

            try:
                # Pass the operation and values to gather a solution of the problem
                problem_output = fcn_math_operation(math_operation.type.tolist()[0], (int(problem_vals[0]), int(problem_vals[1])))
            except:
                # Something went wrong, show an ERROR on the worksheet page
                cv2.rectangle(img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
                cv2.rectangle(clean_output_img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
                textsize = cv2.getTextSize('Error!!', font, 1, 2)[0]
                textX = (img_rgb.shape[1] - textsize[0]) / 2
                textY = (img_rgb.shape[0] + textsize[1]) / 2
                cv2.putText(img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
                cv2.putText(clean_output_img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
                break

            # Paste the answer on the sheet
            cv2.putText(img_rgb, str(int(problem_output)), (int(math_operation.x)+15, int(math_operation.y)+45), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

            cv2.putText(clean_output_img_rgb, str(int(problem_output)), (int(math_operation.x)+15, int(math_operation.y)+45), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

        else:
            # Code for problems that have more than overall two digits
            # First number in math problem
            group_1 = []
            # Second number in math problem
            group_2 = []
            x_loc_vals = numbers_only_sorted[['x']].to_numpy()
            y_loc_vals = numbers_only_sorted[['y']].to_numpy()
            actual_vals = np.array(numbers_only_sorted.type.tolist()).astype(np.int)

            try:
                group_1.append(actual_vals[0])
                top_y_val = y_loc_vals[0]
            except:
                cv2.rectangle(img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
                cv2.rectangle(clean_output_img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
                textsize = cv2.getTextSize('Error!!', font, 1, 2)[0]
                textX = (img_rgb.shape[1] - textsize[0]) / 2
                textY = (img_rgb.shape[0] + textsize[1]) / 2
                cv2.putText(img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
                cv2.putText(clean_output_img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
                break

            # Utilize the sorted x and y positions of the digits to determine the first number and the second number since each digit is individually detected
            curr_val_indx = 0
            group_2_being_filled = 0
            for j in y_loc_vals[1:]:
                if abs(j - top_y_val) <= 3 and group_2_being_filled == 0:
                    curr_val_indx = curr_val_indx + 1
                    group_1.append(actual_vals[curr_val_indx])
                    top_y_val = j
                else:
                    group_2_being_filled = 1
                    curr_val_indx = curr_val_indx + 1
                    group_2 = sum(e * 10 ** i for i, e in enumerate(actual_vals[curr_val_indx:][::-1]))
                    # Just need to run this once, grab the rest of the values and join to create the second number
                    break

            # Combine the individual digits in the first number
            group_1 = sum(e * 10 ** i for i, e in enumerate(group_1[::-1]))

            try:
                problem_output = fcn_math_operation(math_operation.type.tolist()[0], (group_1, group_2))
            except:
                cv2.rectangle(img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
                cv2.rectangle(clean_output_img_rgb, (10, 10), (img_rgb.shape[1], img_rgb.shape[0]), (255,255,255), -1)
                textsize = cv2.getTextSize('Error!!', font, 1, 2)[0]
                textX = (img_rgb.shape[1] - textsize[0]) / 2
                textY = (img_rgb.shape[0] + textsize[1]) / 2
                cv2.putText(img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
                cv2.putText(clean_output_img_rgb, 'Error!!', (int(textX),int(textY)), font, 2, (0,0,255), 3, cv2.LINE_AA)
                break

            # Paste the answer on the sheet
            cv2.putText(img_rgb, str(int(problem_output)), (int(math_operation.x)+15, int(math_operation.y)+45), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

            cv2.putText(clean_output_img_rgb, str(int(problem_output)), (int(math_operation.x)+15, int(math_operation.y)+45), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return img_rgb, clean_output_img_rgb

if __name__ == '__main__':

    img = cv2.imread(args.mathWorksheet)

    img_rgb, clean_output_img_rgb = generate_answer_key(img)

    # Write both the full detection image and answer key image
    cv2.imwrite(args.mathWorksheet[:-4] + '_all_detections.png',img_rgb)
    print('Generated Answer Key in ./MathWorksheets Directory. Happy Grading!!')
    cv2.imwrite(args.mathWorksheet[:-4] + '_answer_key.png',clean_output_img_rgb)
    

