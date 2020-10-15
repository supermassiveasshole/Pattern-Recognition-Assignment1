import { NDArray } from 'vectorious';

export function transpose(mat: NDArray): NDArray {
    if (mat.shape.length < 2) {
      return mat;
    }
  
    let tempRowData = [];
    let columns = mat.shape[1];
    let rows = mat.shape[0];
    for(let i = 0; i < columns; i++) {
      for(let j = 0; j < rows; j++) {
        tempRowData.push(mat.data[i+j*mat.strides[0]]);
      }
    }
    tempRowData.forEach((val, index) => {
      mat.data[index] = val;
    });

    let tmp = mat.shape[0];
    mat.shape[0] = mat.shape[1];
    mat.shape[1] = tmp;
  
    tmp = mat.strides[0];
    mat.strides[0] = mat.strides[1];
    mat.strides[1] = tmp;
  
    return mat;
};