import { NDArray, random } from "vectorious";
import { sigmoid } from "../Maths/sigmoid";
import { reshape } from "vectorious/built/core/reshape";
import { Optimizers } from "../Maths/optimizers";
import { transpose } from "../Maths/transpose";
import { Matrix } from "ml-matrix";

export class LogisticRegression {

    private beta: Matrix; // 这个是我们要优化的参数，使得预测值为sigmoid(beta.T.dot(xHat))

    /**
     * 构造函数
     * @param sampleDim 样本的维度
     */
    constructor(private sampleDim: number) {
        // 两个优化器依赖可以通过依赖注入来获取
        this.beta = Matrix.zeros(sampleDim + 1, 1);
    }

    /**
     * 预测方法
     * @param samples 想要预测的样本
     * @returns 一个Matrix
     */
    public predict(samples: Matrix): Matrix {
        let addOn = new Array(samples.columns).fill(1);
        const samplesHatArr = samples.addRow(samples.rows, addOn);
        let logisticVal = [];
        let predictResults = [];
        for(let i = 0; i < samplesHatArr.columns; i++) {
            const xHat = samplesHatArr.getColumnVector(i);
            logisticVal.push(sigmoid(this.beta.dot(xHat)));
            predictResults.push(
                logisticVal[i] >= 0.5 ? 1 : 0
            );
        }
        return new Matrix([predictResults]);
    }

    /**
     * 使用Newton Method来fit
     * @param trainSet 训练集 
     * @param trainSetTags 训练集标记
     * @param testSet 测试集
     * @param testSetTags 测试集标记
     * @param iters 迭代次数
     * @param learningRate 学习率
     * @param optimizer 要是用的优化器
     * @returns 返回在测试集上的准确率
     */
    public fit(trainSet: Matrix, trainSetTags: Matrix, testSet: Matrix, testSetTags: Matrix, iters: number, learningRate: number, optimizer: Optimizers): [number,number []] {
        // 先将训练集改为增广矩阵xHatMat
        let addOn = new Array(trainSet.columns).fill(1);
        let xHatMat = trainSet.addRow(trainSet.rows, addOn);
        // 求得优化后的beta
        this.beta = optimizer.optimize(this.beta, learningRate, iters, xHatMat, trainSetTags); 
        // 下面测试下在测试集上的性能
        const predictResults = this.predict(testSet); // 获取对测试集的预测结果
        // 统计预测正确的数量
        let numOfCorrectPredictions = 0;
        predictResults.to1DArray().forEach((val: number, index: number) => {
            if(testSetTags.get(index, 0) == val)
                numOfCorrectPredictions++;
        });
        // 计算正确率
        return [numOfCorrectPredictions / testSetTags.rows, predictResults.to1DArray()]
    }

    /**
     * 保存模型
     * @param path 模型保存路径
     */
    public saveModel(path: string): void {
        const fs = require('fs');
        const content = JSON.stringify(this.beta.to2DArray());
        try {
            const data = fs.writeFileSync(path, content);
            //文件写入成功。
        } catch (err) {
            console.error(err);
        }
    }

    /**
     * 读取模型
     * @param path 模型读取路径
     */
    public loadModel(path: string): void {
        const fs = require('fs');
        const data: string = fs.readFileSync(path, 'utf-8');
        this.beta = new Matrix(JSON.parse(data));
        console.log(this.beta.dot(this.beta));
    }
}