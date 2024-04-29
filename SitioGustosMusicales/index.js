// Lista de grupos musicales
const groupsData = {
  "The Beatles": "Rock",
  "Led Zeppelin": "Rock",
  "Pink Floyd": "Rock",
  "Michael Jackson": "Pop",
  "Madonna": "Pop",
  "Britney Spears": "Pop",
  "Eminem": "Hip Hop",
  "Tupac": "Hip Hop",
  "Notorious B.I.G.": "Hip Hop"
};

// Generar la lista de grupos musicales dinámicamente
const groupsListContainer = document.getElementById('groups-list');
for (const group in groupsData) {
  groupsListContainer.innerHTML += `
      <div class="group-container">
          <label for="${group}">${group}</label>
          <input type="number" id="${group}" name="${group}" min="1" max="10">
      </div>
  `;
}

// Función para procesar la información y mostrar el ranking de géneros musicales
document.getElementById('submit-btn').addEventListener('click', async function() {
  const data = [];
  // Recorrer cada grupo y obtener su calificación
  for (const group in groupsData) {
      const rating = parseInt(document.getElementById(group).value);
      if (!isNaN(rating)) {
          data.push({ group: group, rating: rating });
      }
  }

  // Convertir los datos en tensores de TensorFlow.js
  const xs = tf.tensor1d(data.map(d => d.rating));

  // Crear y entrenar el modelo de regresión lineal simple
  const model = createModel();
  await trainModel(model, xs);

  // Realizar predicciones con el modelo
  const predictions = {};
  for (const group in groupsData) {
      const rating = parseInt(document.getElementById(group).value);
      if (!isNaN(rating)) {
          const prediction = model.predict(tf.tensor2d([rating], [1, 1])).dataSync()[0];
          predictions[groupsData[group]] = (predictions[groupsData[group]] || 0) + prediction;
      }
  }

  // Ordenar los géneros por preferencia y mostrar el ranking
  const sortedGenres = Object.keys(predictions).sort((a, b) => predictions[b] - predictions[a]);

  // Mostrar el ranking de géneros musicales al usuario
  const rankingContainer = document.getElementById('ranking');
  rankingContainer.innerHTML = '<h2>Ranking de Géneros Musicales</h2>';
  sortedGenres.forEach((genre, index) => {
      rankingContainer.innerHTML += `<p>${index + 1}. ${genre}</p>`;
  });
});

// Función para crear el modelo de regresión lineal simple
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  model.add(tf.layers.dense({ units: 1, useBias: true }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return model;
}

// Función para entrenar el modelo
async function trainModel(model, xs) {
  const ys = tf.tensor1d(Array.from(Array(xs.shape[0]).keys())); // Usamos un índice como salida
  await model.fit(xs, ys, { epochs: 100 });
}