import { Optimizers } from "./optimizers";
import { Matrix, inverse } from "ml-matrix"

export class NewtonOpt extends Optimizers {

    /**
     * 构造函数
     * @param firstDerivative 一阶导函数
     * @param secondDerivative 二阶导函数
     */
    constructor(private firstDerivative: (x: Matrix, y: Matrix, toOptimize: Matrix) => Matrix,
    private secondDerivative: (x: Matrix, y: Matrix, toOptimize: Matrix) => Matrix) {
        super();
    }

    /**
     * 
     * @param toOptimize 想要优化的值，可以是一个数也可以是一个向量或矩阵
     * @param learningRate 学习率
     * @param iterations 迭代次数
     * @param trainSet 训练集
     */
    optimize(toOptimize: Matrix, learningRate: number, iterations: number, trainSet?: Matrix, tags?: Matrix): Matrix {
        let numOfIter: number = iterations;
        let optimized: Matrix = new Matrix(toOptimize.to2DArray());
        while(numOfIter) {
            const secondDerivative = this.secondDerivative(trainSet, tags, optimized);
            const firstDerivative = this.firstDerivative(trainSet, tags, optimized);
            const secondDerivativeInv = inverse(secondDerivative);
            const multiplied = secondDerivativeInv.mmul(firstDerivative);
            optimized.subtract(multiplied.mul(learningRate));
            numOfIter--;
            console.log(numOfIter+"times of iters left!");
        }
        return optimized;
    }

}