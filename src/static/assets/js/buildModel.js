let objects = [];
let nextId = 1;

function addObject() {
    let type = document.getElementById('type').value;

    // Common field for both types
    let nameField = type === 'planet' ? document.getElementById('name') : document.getElementById('starname');
    let positionField = type === 'planet' ? document.getElementById('position') : document.getElementById('starposition');
    let position = positionField.value;
    let positionNumbers = position.split(',').map(Number);

    if (positionNumbers.length !== 3 || positionNumbers.some(isNaN)) {
        alert("Position must include exactly three numbers, formatted as x,y,z.");
        return; // Stop the function if position is not in the correct format
    }

    if (type === 'planet') {
        let massField = document.getElementById('mass');
        let radiusField = document.getElementById('radius');
        let speedField = document.getElementById('speed');
        let mass = massField.value;
        let radius = radiusField.value;
        let speed = speedField.value;
        let speedNumbers = speed.split(',').map(Number);

        if (!nameField.value || !mass || !radius || !position || !speed) {
            alert("All fields must be filled out to add a planet.");
            return;
        }

        if (speedNumbers.length !== 3 || speedNumbers.some(isNaN)) {
            alert("Speed must each include exactly three numbers, formatted as vx,vy,vz.");
            return;
        }

        mass = parseFloat(mass);
        radius = parseFloat(radius);

        // Create the planet object
        let obj = {
            id: nextId++,
            type: type,
            name: nameField.value,
            mass: mass,
            radius: radius,
            position: position,
            speed: speed,
            color: 'grey'
        };

        objects.push(obj);
        console.log("Added:", obj);
        updateUI(obj);

        // Clearing the fields after adding the object
        nameField.value = '';
        massField.value = '';
        radiusField.value = '';
        positionField.value = '';
        speedField.value = '';

    } else if (type === 'star') {

        if (!nameField.value || !position) {
            alert("All fields must be filled out to add a star.");
            return;
        }

        // Create the star object
        let obj = {
            id: nextId++,
            type: type,
            name: nameField.value,
            position: position,
            color: 'grey'  // Assuming color for stars for now
        };

        objects.push(obj);
        console.log("Added:", obj);
        updateUI(obj);

        // Clearing the fields after adding the object
        nameField.value = '';
        positionField.value = '';
    }
}


function updateUI(obj) {
    const knownStars = ["sun"];
    const knownPlanets = ["earth", "venus", "mercury", "saturn", "mars", "jupiter", "uranus", "neptune"];

    // Determine the image source based on the name and type
    let imageSrc;
    if (obj.type === 'planet' && knownPlanets.includes(obj.name.toLowerCase())) {
        imageSrc = `${obj.name.toLowerCase()}.webp`;
    } else if (obj.type === 'star' && knownStars.includes(obj.name.toLowerCase())) {
        imageSrc = `${obj.name.toLowerCase()}.webp`;  // Specific image for the sun or other specific stars
    } else {
        imageSrc = obj.type === 'planet' ? 'planet.webp' : 'star.webp';  // Default images for unknown planets and stars
    }

    let container = document.querySelector('.inner-transparent-background');
    let newObjectHTML = `
        <div class="object-box">
            <img src="../static/images/${imageSrc}" alt="Object Image" class="object-image">
            <div class="object-parameters">
                <p><span class="parameter-label">Name:</span> ${obj.name}</p>
                ${obj.type === 'planet' ? `
                <p><span class="parameter-label">Mass:</span> ${obj.mass} kg</p>
                <p><span class="parameter-label">Radius:</span> ${obj.radius} km</p>
                <p><span class="parameter-label">Speed:</span> (${obj.speed}) km/s</p>` : `
                `}
                <p><span class="parameter-label">Position:</span> (${obj.position})</p>
            </div>
            <button class="delete-button" onclick="removeObject(${obj.id})">X</button>
        </div>
    `;
    container.innerHTML += newObjectHTML;
}


// <p><span class="parameter-label">Type:</span> ${obj.type}</p>
function removeObject(id) {
    let index = objects.findIndex(obj => obj.id === id);
    if (index > -1) {
        objects.splice(index, 1);  // Remove the object from the array
        document.querySelectorAll('.object-box').forEach((box) => {
            let button = box.querySelector('.delete-button');
            if (button && parseInt(button.getAttribute('onclick').match(/\d+/)[0]) === id) {
                box.remove();  // Remove the object box from the DOM
            }
        });
        console.log("Object removed:", objects);
    }
}



function sendObjects() {
    const url = simulateUrl
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({objects: objects})  // Send objects as JSON
    })
     .then(response => response.json())
    .then(data => {
        console.log("Server response:", data);
        alert('The model has been built! Press continue to proceed.');
    })
    .catch(error => {
        console.error('Error:', error);
         alert('Failed to build model');
    });
}




function toggleFields() {
    var select = document.getElementById('type');
    var planetFields = document.getElementById('planet-fields');
    var starFields = document.getElementById('star-fields');

    console.log('Changing fields, selected value:', select.value); // Debugging output

    if (select.value === 'planet') {
        console.log('Displaying planet fields');
        planetFields.style.display = 'block';  // Show planet fields
        starFields.style.display = 'none';     // Hide star fields
    } else if (select.value === 'star') {
        console.log('Displaying star fields');
        planetFields.style.display = 'none';   // Hide planet fields
        starFields.style.display = 'block';    // Show star fields
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var typeSelect = document.getElementById('type');
    if (typeSelect) {
        typeSelect.addEventListener('change', toggleFields); // Attach event listener to dropdown
    } else {
        console.log('Dropdown #type not found'); // Error if not found
    }
    toggleFields(); // Call to set initial visibility based on the default selected option

     fetch('/get_planets')
    .then(response => response.json())
    .then(data => {
        planetsData = data;
        console.log("Planet data loaded:", planetsData);

        // Ensure the 'name' input is available and attach event listener for autofill
        var nameField = document.getElementById('name');
        if (nameField) {
            nameField.addEventListener('input', function() {
                autofillPlanetData(this.value);
            });
            console.log("Event listener attached to name field.");
        } else {
            console.log("Name field not found.");
        }
    })
    .catch(error => console.error('Error loading planet data:', error));
});

let planetsData = [];


function toggleFields() {
    var select = document.getElementById('type');
    var planetFields = document.getElementById('planet-fields');
    var starFields = document.getElementById('star-fields');

    if (select && select.value === 'planet') {
        planetFields.style.display = 'block';
        starFields.style.display = 'none';
    } else if (select && select.value === 'star') {
        planetFields.style.display = 'none';
        starFields.style.display = 'block';
    }
}

function autofillPlanetData(planetName) {
    console.log("Trying to autofill for:", planetName);
    if (!planetsData) {
        console.log("Planets data not loaded yet.");
        return;
    }
    const planet = planetsData.find(p => p.name.toLowerCase() === planetName.toLowerCase());
    if (planet) {
        document.getElementById('mass').value = planet.mass;
        document.getElementById('radius').value = planet.radius / 1000; // Convert meters to kilometers
        document.getElementById('position').value = planet.location.join(',');
        document.getElementById('speed').value = planet.speed.join(',');
    } else {
        console.log("No planet found for:", planetName);
    }
}