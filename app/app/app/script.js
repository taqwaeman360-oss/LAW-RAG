const API_URL = "http://127.0.0.1:8000"; // Replace with your Render/Railway backend URL after deployment

async function askQuestion() {
    const questionInput = document.getElementById("user-input").value;
    if (!questionInput) return;

    try {
        const response = await fetch(`${API_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: questionInput })
        });

        const data = await response.json();
        
        // Display response & cited sources
        console.log("Answer:", data.answer);
        console.log("Sources:", data.sources);
    } catch (error) {
        console.error("Error communicating with API:", error);
    }
}
