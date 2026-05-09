let sidebarOpen = false;

function toggleSidebar() {
    const sidebar = document.querySelector(".sidebar");
    
    if (!sidebarOpen) {
        sidebar.style.width = "250px";
        sidebarOpen = true;
    } else {
        sidebar.style.width = "0";
        sidebarOpen = false;
    }
}

// Close sidebar if user clicks outside of it
window.onclick = function(event) {
    const sidebar = document.querySelector(".sidebar");
    const menuBtn = document.querySelector(".menu-btn");
    if (sidebarOpen && !sidebar.contains(event.target) && event.target !== menuBtn) {
        sidebar.style.width = "0";
        sidebarOpen = false;
    }
}