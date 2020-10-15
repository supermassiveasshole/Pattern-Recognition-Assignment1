namespace Shuffle {
  function shuffle(array) {
    var currentIndex = array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {

      // Pick a remaining element...
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex -= 1;

      // And swap it with the current element.
      temporaryValue = array[currentIndex];
      array[currentIndex] = array[randomIndex];
      array[randomIndex] = temporaryValue;
    }

    return array;
  }

  const fs = require('fs');
  const readline = require('readline');
  let arr = [];

  let data: string = fs.readFileSync('./iris.data', 'utf8');
  arr = data.split('\n');
  shuffle(arr);
  fs.writeFileSync('./iris_processed.json', JSON.stringify(arr));
}
