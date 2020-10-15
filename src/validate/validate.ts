const { Matrix } = require('ml-matrix');
import LogisticRegression from './logreg'

let tagMap = new Map();
tagMap.set("Iris-setosa", 1);
tagMap.set("Iris-versicolor", 0);

const fs = require('fs');
let arr = [];
let data: string = fs.readFileSync('./iris_processed.json', 'utf8');
arr = JSON.parse(data);

let samples = [];
let sampleTags = [];
arr.forEach((val: string, index: number) => {
    let tokens = val.split(',', 5);
    samples.push([Number(tokens[0]), Number(tokens[1])]);
    sampleTags.push([tagMap.get(tokens[4])]);
});

let X = new Matrix(samples.slice(0, samples.length - 20));
let Y = Matrix.columnVector(sampleTags.slice(0, sampleTags.length - 20));

let Xtest = new Matrix(samples.slice(samples.length - 20, samples.length));
let Ytest = Matrix.columnVector(sampleTags.slice(samples.length - 20, sampleTags.length));

// We will train our model.
const logreg = new LogisticRegression({ numSteps: 100, learningRate: 1e-2 });
logreg.train(X, Y);

// We try to predict the test set.
const finalResults = logreg.predict(Xtest) as Array<number>;
console.log(finalResults);

let num = 0;
finalResults.forEach((val, index) => {
    if(sampleTags.slice(samples.length - 20, sampleTags.length)[index] == val)
        num++;
});
console.log(num/finalResults.length);