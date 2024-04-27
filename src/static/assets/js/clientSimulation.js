function start1() {
    fetch('/buildModel2')
    .then(response => response.text())
    .then(data => alert(data))
    .catch(error => console.error('Error starting simulation:', error));
}