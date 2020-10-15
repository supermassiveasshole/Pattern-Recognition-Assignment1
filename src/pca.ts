import * as PCA from 'pca-js/pca.min.js';
import * as iris from '../iris_processed.json';

let irisPrePCA = iris as string[]
let irisPrePCAProcessed: {
    data: number[][];
    tags: number[];
};

let tagMap = new Map();
tagMap.set("Iris-setosa", 1);
tagMap.set("Iris-versicolor", 0);

let data: number[][] = [];
let tags: number[] = [];

irisPrePCA.forEach((val, index) => {
    let tokens = val.split(',', 5);
    data.push([Number(tokens[0]), Number(tokens[1]), Number(tokens[2]), Number(tokens[3])]);
    tags.push(tagMap.get(tokens[4]));
});
irisPrePCAProcessed = {
    data: data,
    tags: tags
};

var vectors = PCA.getEigenVectors(data);

var adData = PCA.computeAdjustedData(irisPrePCAProcessed.data,vectors[0],vectors[1]);

irisPrePCAProcessed.data = adData.adjustedData;

const fs = require('fs');
fs.writeFileSync('./iris_processed_PCA.json', JSON.stringify(irisPrePCAProcessed));
