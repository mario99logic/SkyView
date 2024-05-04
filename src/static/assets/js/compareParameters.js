function calculate_results() {
    fetch('/compute-results', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok. Status: ' + response.status);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            console.log('Velocity:', data.velocity);
            console.log('New Position:', data.position);

            document.getElementById('velocity-value').innerText = `(${data.velocity})` ;
            document.getElementById('position-value').innerText = `(${data.position})` ;
            document.getElementById('velocity-uncertainty').innerText = '±0.01'; // Example uncertainty
            document.getElementById('position-uncertainty').innerText = '±10'; // Example uncertainty
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to fetch results: ' + error.message);
    });
}


