// 规定模块中所有的向量都是列向量，向量和矩阵在这里都是“二维”的。

import { LogisticRegression } from "./logistic/logistic-regression";
import { NewtonOpt } from "./Maths/newtonOpt";
import { firstLogisticDerivative, secondLogisticDerivative } from "./logistic/derivatives";
import { GradientDecOpt } from "./Maths/gradientDecOpt";
import { Matrix } from 'ml-matrix'

let tagMap = new Map();
tagMap.set("Iris-setosa", 1);
tagMap.set("Iris-versicolor", 0);

const fs = require('fs');
let arr = [];
let data: string=fs.readFileSync('./iris_processed.json','utf8');
arr = JSON.parse(data);

let samples = [];
let sampleTags = [];
arr.forEach((val: string, index: number) => {
    let tokens = val.split(',', 5);
    samples.push([Number(tokens[0]), Number(tokens[1])]);
    sampleTags.push([tagMap.get(tokens[4])]);
});

let trainSet = new Matrix(samples.slice(0, samples.length - 20)).transpose();
let trainSetTags = new Matrix(sampleTags.slice(0, sampleTags.length - 20));
let testSet = new Matrix(samples.slice(samples.length - 20, samples.length)).transpose();
let testSetTags = new Matrix(sampleTags.slice(samples.length - 20, sampleTags.length));


// 我就手动注入了
let gradientDecOpt = new GradientDecOpt(firstLogisticDerivative);
let newtonOpt = new NewtonOpt(firstLogisticDerivative, secondLogisticDerivative);
let logisticModel = new LogisticRegression(2);
const [rate, predict] = logisticModel.fit(trainSet, trainSetTags, testSet, testSetTags, 2000, 1e-2, gradientDecOpt);
console.log(rate);
console.log(predict);
logisticModel.saveModel('./myLogisticRegression1.json');