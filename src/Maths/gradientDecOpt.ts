import { Optimizers } from "./optimizers";
import { NDArray } from "vectorious";
import { Matrix } from "ml-matrix"

export class GradientDecOpt extends Optimizers {

    /**
     * 构造函数
     * @param gradient 梯度
     * @param loss_func 损失函数
     */
    constructor(private gradient: (trainSet: Matrix, tags: Matrix, toOptimize: Matrix) => Matrix,
    private loss_func?: (trainSet: Matrix, tags: Matrix, toOptimize: Matrix) => Matrix) {

        // 前两个参数可注入
        super();
    }
    /**
     * 优化方法
     * @param toOptimize 要优化的参数
     * @param learningRate 学习率
     * @param iterations 迭代次数
     * @param trainSet 可能用到的训练集
     * @param tags 可能用到的训练集标记
     */
    optimize(toOptimize: Matrix, learningRate: number, iterations: number, trainSet?: Matrix, tags?: Matrix): Matrix {
        // 先写的简单点，不返回每轮的情况
        let itersLeft = iterations;
        let optimized = new Matrix(toOptimize.to2DArray());
        while(itersLeft) {
            let gradient = this.gradient(trainSet, tags, optimized);
            optimized.subtract(gradient.mul(learningRate));
            itersLeft--;
            console.log(itersLeft+" times of iters left!");
        }
        return optimized;
    }
    

}