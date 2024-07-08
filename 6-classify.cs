// Classifies images with ONNX model and returns predicted labels.
//
// dotnet add package Microsoft.ML.OnnxRuntime
// dotnet add package SixLabors.ImageSharp
// dotnet run 6-classify.cs

// Output:
//
// Image negative-question.png most likely belongs to class option-a.
// Image no-entry.png most likely belongs to class no-entry.
// Image no-park-question.jpg most likely belongs to class no-park.
// Image no-park.png most likely belongs to class no-park.
// Image no-stop-question.jpg most likely belongs to class no-stop.
// Image no-stop.png most likely belongs to class no-stop.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Text.Json;
using Image = SixLabors.ImageSharp.Image;


var modelPath = "models/model.onnx";
var imagesFolder = Directory.GetFiles("images/test/example");
var classNames = JsonSerializer.Deserialize<string[]>(File.ReadAllText("models/class_names.json"));

foreach (var imagePath in imagesFolder)
{
	// Copy over image pixels to a tensor object (vector space).
	var image = Image.Load<Rgb24>(imagePath);
	image.Mutate(ctx => ctx.Resize(new Size(224, 224)));
	var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
	for (int y = 0; y < 224; y++)
	{
		for (int x = 0; x < 224; x++)
		{
			tensor[0, 0, y, x] = image[x, y].R / 255f;
			tensor[0, 1, y, x] = image[x, y].G / 255f;
			tensor[0, 2, y, x] = image[x, y].B / 255f;
		}
	}

	// Load the model and run the inference on tensor input.
	using var session = new InferenceSession(modelPath);
	using var results = session.Run(new List<NamedOnnxValue>
	{
		NamedOnnxValue.CreateFromTensor("input", tensor)
	});

	// Get the output and print the predicted class.
	var output = results.First().AsEnumerable<float>().ToArray();
	var predicted_class = output.ToList().IndexOf(output.Max());
	Console.WriteLine($"Image {Path.GetFileName(imagePath)} most likely belongs to class {classNames![predicted_class]}.");
}