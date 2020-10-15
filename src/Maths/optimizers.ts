import { NDArray } from "vectorious";
import { Matrix } from 'ml-matrix';

export abstract class Optimizers {
    /**
     * 优化函数
     * @param toOptimize 想要优化的值，可以是一个数也可以是一个向量或矩阵
     * @param learningRate 学习率
     * @param iterations 迭代次数
     * @param x 训练集
     */
    abstract optimize(toOptimize: Matrix, learningRate: number, iterations: number, trainSet?: Matrix, tags?: Matrix): Matrix;
}