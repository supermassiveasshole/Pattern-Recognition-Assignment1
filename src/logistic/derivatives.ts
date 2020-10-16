/**
 * 定义了一些logistic 回归进行参数优化所需要的导函数
 */
import { p1 } from "./probability";
import { Matrix } from 'ml-matrix';


export function  firstLogisticDerivative(xHatMat: Matrix, y: Matrix, beta: Matrix): Matrix {
    // 每个样本是个列向量
    let num = xHatMat.columns;
    let dim = xHatMat.rows;
    let result = Matrix.zeros(dim, 1);
    for(let i = 0; i < num ; i++) {
        // 对每个样本
        result.add(xHatMat.getColumnVector(i).mul(y.get(i,0) - p1(xHatMat.getColumnVector(i), beta)));
    }
    result.neg();
    return result;
}

export function secondLogisticDerivative(xHatMat: Matrix, y: Matrix, beta: Matrix): Matrix {
    // 每个样本是个列向量
    let num = xHatMat.columns;
    let dim = xHatMat.rows;
    let result = Matrix.zeros(dim, dim);
    for(let i = 0; i < num; i++) {
        const p1Result = p1(xHatMat.getColumnVector(i), beta);
        const scalar = p1Result*(1 - p1Result);
        const matrix = xHatMat.getColumnVector(i).mmul(xHatMat.getColumnVector(i).transpose());
        result.add(matrix.mul(scalar)); 
    }
    return result;
}
