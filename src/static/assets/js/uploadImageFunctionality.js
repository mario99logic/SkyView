// File: static/js/drag-and-drop.js

document.addEventListener("DOMContentLoaded", function () {
    var dropArea = document.getElementById('drop-area');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        var dt = e.dataTransfer;
        var files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        ([...files]).forEach(uploadFile);
    }

    function uploadFile(file) {
        var url = '/'; // Endpoint URL
        var formData = new FormData();
        formData.append('image', file);

        fetch(url, {
            method: 'POST',
            body: formData
        })
        .then(() => { alert('File uploaded successfully'); })
        .catch(() => { alert('Upload failed'); });
    }
});


// File: static/js/drag-and-drop.js

// Function to simulate clicking on the hidden file input
function triggerFileInput() {
    document.getElementById('fileInput').click();
}


function uploadFiles(files) {
    var url = '/upload'; // Endpoint URL where the file should be uploaded
    var formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]); // Using 'images' as the key for multiple files
    }

    fetch(url, {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(text => {
        alert('Upload successful: ' + text);
    })
    .catch(error => {
        console.error('Error during upload:', error);
        alert('Upload failed');
    });
}



function previewImage(files) {
    var imageContainer = document.getElementById('imagePreviewContainer');
    var uploadIcon = document.getElementById('uploadIcon'); // Get the upload icon
    var uploadText = document.querySelector('.upload-text'); // Get the upload text element

    // Hide the upload icon and the upload text as soon as a file is selected
    uploadIcon.classList.add('hidden');
    uploadText.classList.add('hidden'); // Add 'hidden' class to the upload text


    // Loop through each file selected
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        if (file.type.match('image.*')) {
            var reader = new FileReader();
            reader.onload = (function(theFile) {
                return function(e) {
                    var span = document.createElement('span');
                    span.classList.add('image-preview-span'); // Add class for styling
                    span.innerHTML = '<img class="thumb" src="' + e.target.result + '" title="' + escape(theFile.name) + '"/><br><span class="image-title">' + escape(theFile.name) + '</span>';
                    imageContainer.appendChild(span); // Append the new image span next to previous ones
                };
            })(file);
            reader.readAsDataURL(file);
        }
    }
}


