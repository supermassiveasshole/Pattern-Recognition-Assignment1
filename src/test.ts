import { Matrix } from 'ml-matrix';

let A = new Matrix([[1], [2]]);
let B = new Matrix(A.to2DArray()).transpose();

console.log(A.mmul(B));