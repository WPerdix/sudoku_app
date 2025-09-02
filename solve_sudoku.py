import numpy as np
import argparse
import cv2

from PIL import Image

from src.sudoku_picture import SudokuPicture
from src.dlx import Sudoku


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Solve a sudoku based on a picture.')
    parser.add_argument('-p', '--path', type=str, default='./sudoku.jpg', required=False, help='path to sudoku image file')
    parser.add_argument('-r', '--resolution', type=int, default=600, required=False, help='display resolution of shortest side')
    
    args = parser.parse_args()
 
    path = args.path
    resolution = args.resolution
    
    solver = SudokuPicture()
    
    picture = cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_BGR2RGB)

    new_shape = (600, int(np.round(picture.shape[0] / picture.shape[1] * 600))) if picture.shape[0] > picture.shape[1] else (int(np.round(picture.shape[1] / picture.shape[0] * 600)), 600)
    cv2.imshow("sudoku", cv2.resize(picture, new_shape))

    solved_sudoku = solver.solve(picture)
    
    if solved_sudoku is None:
        print('Either no sudoku was detected or not enough lines were detected.')
    elif type(solved_sudoku) == str:
        sudoku = Sudoku(solved_sudoku)
        print(sudoku.print_state(solved_sudoku))
    else:
        j = Image.fromarray(solved_sudoku)
        solved_sudoku = cv2.resize(solved_sudoku, new_shape)
        
        cv2.imshow("solved", solved_sudoku)
    cv2.waitKey(0)

