// uni_mapper.js
window.getUniLogo = function(teamNameField) {
    if (!teamNameField || !teamNameField.includes('|')) return null;
    const uniPrefix = teamNameField.split('|')[0].trim().toLowerCase();
    
    const exactFiles = {
        "uom": "uom.png",
        "uok": "uok.png",
        "iit": "IIT.png",
        "java": "java.png",
        "sliit": "sliit.png",
        "uor": "uor.png",
        "uoj": "uoj.png",
        "uop": "uop.png",
        "nibm": "nibm.png",
        "cinec": "cinec.png"
    };

    if (exactFiles[uniPrefix]) return `images/uni/${exactFiles[uniPrefix]}`;
    return `images/uni/${uniPrefix}.png`; // fallback
};

window.cleanTeamName = function(teamNameField) {
    if (!teamNameField || !teamNameField.includes('|')) return teamNameField;
    return teamNameField.split('|')[1].trim();
};
