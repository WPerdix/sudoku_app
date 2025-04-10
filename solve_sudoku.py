import numpy as np
import argparse
import cv2

from PIL import Image

from src.sudoku_picture import SudokuPicture


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Get the factorial of the integer.')
    parser.add_argument('-p', '--path', type=str, default='./sudoku.jpg', required=False, help='path to sudoku image file')
    
    args = parser.parse_args()
 
    path = args.path
    
    solver = SudokuPicture()
    
    picture = np.array(Image.open(path))

    cv2.imshow("sudoku", picture)

    solved_sudoku = solver.solve(picture)
    
    cv2.imshow("solved", solved_sudoku)
    cv2.waitKey(0)

