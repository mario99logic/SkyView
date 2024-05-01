function startSimulation() {
    fetch('/start-simulation')
    .then(response => response.text())
    .then(data => alert(data))
    .catch(error => console.error('Error starting simulation:', error));
}

function startSimulation2() {
    fetch('/start-simulation2')
    .then(response => response.text())
    .then(data => alert(data))
    .catch(error => console.error('Error starting simulation:', error));
}