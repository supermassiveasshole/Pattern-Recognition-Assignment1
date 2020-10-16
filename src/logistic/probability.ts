import { Matrix } from "ml-matrix";

// this function means returning the probability of x being a negative sample
export function p0(xHat: Matrix, beta: Matrix): number {
    return 1/(1+Math.exp(beta.dot(xHat)));
}

// this function means returning the probability of x being a negative sample
export function p1(xHat: Matrix, beta: Matrix): number {
    return 1 - p0(xHat, beta);
}