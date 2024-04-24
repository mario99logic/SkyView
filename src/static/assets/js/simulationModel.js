function startSimulation() {
    fetch('/start-simulation')
    .then(response => response.text())
    .then(data => alert(data))
    .catch(error => console.error('Error starting simulation:', error));
}