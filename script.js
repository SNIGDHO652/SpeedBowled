const videoUpload = document.getElementById('video-upload');
const video = document.getElementById('video');
const startBtn = document.getElementById('start-btn');
const pauseBtn = document.getElementById('pause-btn');
const calculateBtn = document.getElementById('calculate-btn');
const colorPicker = document.getElementById('color-picker');
const selectedColorSpan = document.getElementById('selected-color');

// Countdown elements
const countdownDiv = document.createElement('div');
countdownDiv.id = 'countdown';
countdownDiv.style.display = 'none';
countdownDiv.innerHTML = `<p>Calculating... <span id="timer">60</span> seconds remaining.</p>`;
document.body.appendChild(countdownDiv);

let selectedHSV = null;

// Convert RGB to HSV
function rgbToHsv(r, g, b) {
    r /= 255, g /= 255, b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const delta = max - min;
    let h = 0, s = 0, v = max;

    if (delta > 0) {
        s = delta / max;
        if (max === r) h = (g - b) / delta + (g < b ? 6 : 0);
        else if (max === g) h = (b - r) / delta + 2;
        else if (max === b) h = (r - g) / delta + 4;
        h /= 6;
    }
    return [Math.round(h * 180), Math.round(s * 255), Math.round(v * 255)];
}

// Button Highlight Effect
function highlightButton(button) {
    const buttons = document.querySelectorAll('.controls button');
    buttons.forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');
}

// Update selected color
colorPicker.addEventListener('input', () => {
    const hex = colorPicker.value;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    selectedHSV = rgbToHsv(r, g, b);
    selectedColorSpan.textContent = `Selected Color: ${hex}`;
});

// Upload Video
videoUpload.addEventListener('change', () => {
    const file = videoUpload.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        video.src = url;
        startBtn.disabled = false;
        pauseBtn.disabled = false;
        calculateBtn.disabled = false;
    }
});

// Start Video
startBtn.addEventListener('click', () => {
    video.play();
    highlightButton(startBtn);
});

// Pause Video
pauseBtn.addEventListener('click', () => {
    video.pause();
    highlightButton(pauseBtn);
});

// Calculate Speeds with Countdown
calculateBtn.addEventListener('click', async () => {
    const formData = new FormData();
    formData.append('video', videoUpload.files[0]);
    if (selectedHSV) formData.append('color', JSON.stringify(selectedHSV));

    let timeLeft = 60;
    const timerElement = document.getElementById('timer');

    countdownDiv.style.display = 'block'; // Show countdown
    highlightButton(calculateBtn);

    const countdown = setInterval(() => {
        timeLeft -= 1;
        timerElement.textContent = timeLeft;

        if (timeLeft <= 0) {
            clearInterval(countdown);
            countdownDiv.style.display = 'none';
            alert("Please try again later with the color of the ball selected.");
        }
    }, 1000);

    try {
        const response = await fetch('/process_video', { method: 'POST', body: formData });
        const data = await response.json();

        clearInterval(countdown);
        countdownDiv.style.display = 'none'; // Hide countdown

        document.getElementById('max-speed').textContent = data.max.toFixed(2);
        document.getElementById('min-speed').textContent = data.min.toFixed(2);
        document.getElementById('avg-speed').textContent = data.avg.toFixed(2);

        // Render Chart
        new Chart(document.getElementById('speedChart'), {
            type: 'line',
            data: {
                labels: data.speeds.map((_, i) => i + 1),
                datasets: [{
                    label: 'Speed (km/h)',
                    data: data.speeds,
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    borderColor: '#ff4500',
                    borderWidth: 2,
                    pointRadius: 2, // Small points for readability
                    pointBackgroundColor: '#ff4500',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // Allow manual sizing
                aspectRatio: 2, // Force smaller ratio (width/height)
                scales: {
                    x: { 
                        grid: { color: '#ddd' }, 
                        ticks: { font: { size: 2 } } 
                    },
                    y: { 
                        grid: { color: '#ddd' }, 
                        ticks: { font: { size: 2 } } 
                    }
                },
                plugins: {
                    legend: { display: true, labels: { font: { size: 2 } } }
                }
            }
        });
        
    } catch (error) {
        clearInterval(countdown);
        countdownDiv.style.display = 'none';
        console.error(error);
        alert("An error occurred. Please try again.");
    }
});
