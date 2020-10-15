# Pattern-Recognition-Assignment1

## How to run this project
- Step1: Download this project or just clone it via git clone. the repository is [here](https://github.com/supermassiveasshole/Pattern-Recognition-Assignment1.git)

- Step2: Here are some runtime environment you should already have installed before running this project. All of these should be prepared if you don't have them all installed correctly.
  - [node.js](https://nodejs.org) ( npm should be installed along with node.js )
  - [typescript](https://www.typescriptlang.org/download) ( We truely recommend installing typescript via npm: `npm i typescript -g` )
  - [ts-node](https://www.npmjs.com/package/ts-node) may have access to simplify the procedure running this project. However, run compiled js files on node.js is always a good solution.
  - [Angular cli](https://angular.io) `npm i -g @angular/cli` is also required if you want to run the data visualization project inside the folder 'LogisticRegressionVisualization'. We also provide a compiled version of this project which you can run it in browsers like Chrome. But a local server is required instead( We recommend the live server extension for vscode ).

- step3: Compile this project
  - Before compile this project, you must run `npm i` to install all the dependencies which this project use. 
  - Use the command `tsc` to compile all `.ts` files into `.js` files. All the js file will be waiting for you inside the 'dist' folder.
  - Alternatively, you could leave this job to ***ts-node*** and skip this step.

- step4: Run this project
  
  To run this project use the command `node dist/index.js` or use `ts-node src/index.ts`. Use the method whichever you like to run this project. 

## Structure of this project
- `src`

  All the source files are in this folder.
  - `index.ts` is the entrance of the program. We load iris data and train our logistic regression model here. Besides, the model will be saved at `myLogisticRegression1.json`. You can load it to do prediction next time.
  - shuffle.ts shuffles the iris data and saves it in `iris_processed.json`.
  - Just ignore `test.ts`.
  - The `validate` folder includes a validation program to test whether our program is running correctly.
  - The `maths` folder contains the mathematic tools we used to implement the logistic regression model including optimizers and probablity functions.
  - The `logistic` folder is the very place where our logistic regression algrithm lies.
- `dist`
  - This folder contains all of the compiled `.js` files.
- `LogisticRegressionVisualization`
  - This folder is a project which is a undergoing logistic regression visualization application.
  - For more detailed information, reference the `README.md` file inside this folder.
- Others
  - Just leave them alone.