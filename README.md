<!DOCTYPE html>
<html>

<head>
    <title>Dota2ML-Predict-Winning-Team</title>
</head>

<body>
    <h1>Dota2ML-Predict-Winning-Team</h1>

    <h2>Introduction</h2>
    <p>Dota 2 basic ML to select a winning team based on heros. Very Basic model, was used for ML learning.</p>

    <h2>Scripts</h2>
    <p>The following scripts are included:</p>
    <ul>
        <li><code>GETCLEANDATA-heropicksmodel.py</code> - captures 100 items from the opendota API.</li>
        <li><code>RFC or SVMprediction.py</code> - predicts a winner based on the heros you enter and from the data in your mongoDB.</li>
    </ul>

    <h2>Getting Started</h2>
    <p>Before running the scripts, you will need to:</p>
    <ol>
        <li>Run the Docker Compose file to allow it to spin up a MongoDB.</li>
        <li>Create a database called <code>dota2</code>. Once you have done this, the script to collect data will automatically create the tables/collections required.</li>
    </ol>

    <h2>Future Plans</h2>
    <p>Future plans for this code include:</p>
    <ul>
        <li>Creating a basic web front-end allowing people to pick the heros and work out best hero matchups or predictions on who would win.</li>
        <li>Adding a range of features, such as but not limited to MMR and Gold Difference, for the model to learn and get a much better understanding of how the game works.</li>
    </ul>

    <p>Any feedback or ideas for changes are welcome.</p>
</body>

</html>
