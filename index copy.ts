// 规定模块中所有的向量都是列向量，向量和矩阵在这里都是“二维”的。

import { LogisticRegression } from "./src/logistic/logistic-regression";
import { NewtonOpt } from "./src/Maths/newtonOpt";
import { firstLogisticDerivative, secondLogisticDerivative } from "./src/logistic/derivatives";
import { GradientDecOpt } from "./src/Maths/gradientDecOpt";
import { NDIter, NDArray } from "vectorious";

type Sample = {
    input: number[];
    output: number[];
}

// 先读样本
let mnist = require('mnist');

// 获取训练集和测试集
let set = mnist.set(8000, 200);
let trainSamples: Sample[] = set.training;
let testSamples: Sample[] = set.test;

// 初始化logistic回归数据集
let x1 = [];
let y1 = [];
let x2 = [];
let y2 = [];
trainSamples.forEach((val: Sample, index: number) => {
    x1.push(val.input);
    y1.push([val.output.findIndex(val => val == 1) >= 5 ? 1 : 0]);
});
testSamples.forEach((val: Sample, index: number) => {
    x2.push(val.input);
    y2.push([val.output.findIndex(val => val == 1) >= 5 ? 1 : 0]);
});
let trainSet = new NDArray(x1).transpose();
let trainSetTags = new NDArray(y1);
let testSet = new NDArray(x2).transpose();
let testSetTags = new NDArray(y2);

// 我就手动注入了
let newtonOpt = new NewtonOpt(firstLogisticDerivative,secondLogisticDerivative);
let gradientDecOpt = new GradientDecOpt();
let logisticModel = new LogisticRegression(newtonOpt, gradientDecOpt, 784);
const rate = logisticModel.fitWithNewtonMethod(trainSet, trainSetTags, testSet, testSetTags, 1, 1);
console.log(rate);
logisticModel.saveModel('./myLogisticRegression.json');