// https://github.com/grensen/multi-core_demo

System.Action<string> print = System.Console.WriteLine;

#if DEBUG
System.Console.WriteLine("Debug mode is on, switch to Release mode\n");
#endif 

print("\nBegin Multi-Core neural network demo\n"); 

// get data
AutoData d = new(@"C:\mnist\");

// define neural network 
int[] network    = { 784, 100, 100, 10 }; 
var LEARNINGRATE = 0.0005f;
var MOMENTUM     = 0.67f;
var EPOCHS       = 50;
var BATCHSIZE    = 1;
var FACTOR       = 0.99f;

RunDemo(d, network, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

print("\nEnd Multi-Core neural network demo");

static void RunDemo(AutoData d, int[] network, float LEARNINGRATE, float MOMENTUM, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    System.Console.WriteLine("NETWORK      = " + string.Join("-", network));
    System.Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    System.Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    System.Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    System.Console.WriteLine("EPOCHS       = " + EPOCHS);
    System.Console.WriteLine("FACTOR       = " + FACTOR + "\n");

    Net singleNet = new(network), multiNet = new(network);

    System.Console.WriteLine("Start reproducible Single-Core training");
    var sTime = RunSingleNet(d, singleNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
    System.Console.WriteLine("\nStart non reproducible Multi-Core training");
    var mTime = RunMultiNet(d, multiNet, 60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);
   
    System.Console.WriteLine("\nTotal time = " + (sTime + mTime).ToString("F2") + "s");
    System.Console.WriteLine("Single time = " + sTime.ToString("F2") + "s" + " Multi time = " + mTime.ToString("F2") + "s");   
    System.Console.WriteLine("Multi-Core was " + (sTime / mTime).ToString("F2") + " times faster than Single-Core");
}
static void PrintInfo(string str ,int correct, int all, DateTime elapsed)
{
    System.Console.WriteLine(str + " accuracy = " + (correct * 100.0 / all).ToString("F2")
        + " correct = " + correct + "/" + all + " time = " + ((DateTime.Now - elapsed).TotalMilliseconds / 1000.0).ToString("F2") + "s");
}
static float RunSingleNet(AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    DateTime elapsed = DateTime.Now;
    RunTrainingSingle(elapsed, d, neural, len, lr, mom, FACTOR, EPOCHS, BATCHSIZE);
    int correct = RunTestSingle(d, neural, 10000);
    PrintInfo("Test", correct, 10000, elapsed);
    return (float)((DateTime.Now - elapsed).TotalMilliseconds / 1000.0f);
}
static float RunMultiNet(AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    DateTime elapsed = DateTime.Now;
    RunTrainingMulti(elapsed, d, neural, len, lr, mom, FACTOR, EPOCHS, BATCHSIZE);
    int correct = RunTestMulti(d, neural, 10000);
    PrintInfo("Test", correct, 10000, elapsed);
    return (float)((DateTime.Now - elapsed).TotalMilliseconds / 1000.0f);
}
static int RunTestSingle(AutoData d, Net neural, int len)
{
    int correct = 0;
    for (int x = 0; x < len; x++) // test
        correct += EvalTest(x, d.labelsTest[x], d.samplesTest, neural) ? 1 : 0;
    return correct;
}
static int RunTestMulti(AutoData d, Net neural, int len)
{    
    bool[] c = new bool[len];
    System.Threading.Tasks.Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, len), range =>
    {
        for (int x = range.Item1; x < range.Item2; x++)
            c[x] = EvalTest(x, d.labelsTest[x], d.samplesTest, neural);
    });
    int correct = 0;
    for (int i = 0; i < c.Length; i++) if (c[i]) correct++;
    return correct;
}
static void RunTrainingSingle(DateTime elapsed, AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int batchSize = 100)
{
    float[] delta = new float[neural.weights.Length];
    for (int epoch = 0, B = len / batchSize; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {
        int correct = 0; // output neurons      
        for (int b = 0; b < B; b++)
        {
            for (int x = b * batchSize, X = x + batchSize; x < X; x++)
                if (EvalAndTrain(x, d.labelsTraining[x], d.samplesTraining, neural, delta)) correct++;
            Update(neural.weights, delta, lr, mom);
        }
        if ((epoch + 1) % 10 == 0) PrintInfo("Epoch = " + (1 + epoch), correct, B * batchSize, elapsed);
    }
}
static void RunTrainingMulti(DateTime elapsed, AutoData d, Net neural, int len, float lr, float mom, float FACTOR, int EPOCHS, int batchSize = 100)
{
    float[] delta = new float[neural.weights.Length];
    for (int epoch = 0, B = len / batchSize; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {
        bool[] c = new bool[len];
        for (int b = 0; b < B; b++)
        {
            System.Threading.Tasks.Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(b * batchSize, (b + 1) * batchSize), range =>
            {
                for (int x = range.Item1; x < range.Item2; x++)
                    c[x] = EvalAndTrain(x, d.labelsTraining[x], d.samplesTraining, neural, delta);
            });
            Update(neural.weights, delta, lr, mom);
        }
        int correct = 0;
        for (int i = 0; i < len; i++) if (c[i]) correct++;
        if ((epoch + 1) % 10 == 0) PrintInfo("Epoch = " + (1 + epoch), correct, B * batchSize, elapsed);
    }
}
static bool EvalAndTrain(int x, byte t, byte[] samples, Net neural, float[] delta)
{
    Span<float> neuron = new float[neural.neuronLen];
    int p = Eval(x, samples, neural, neuron);
    if (neuron[neural.neuronLen - neural.net[^1] + t] < 0.99)
        BP(neural, neuron, delta, t);
    return p == t;

    static void BP(Net neural, Span<float> neuron, float[] delta, int target)
    {
        Span<float> gradient = new float[neuron.Length];
        int inputHidden = neuron.Length - neural.net[neural.layers]; // size of input and hidden neurons

        // output error gradients, hard target as 1 for its class
        for (int r = inputHidden, p = 0; r < neuron.Length; r++, p++)
            gradient[r] = target == p ? 1 - neuron[r] : -neuron[r];

        for (int i = neural.layers - 1, j = inputHidden, k = neuron.Length, m = neural.weights.Length; i >= 0; i--)
        {
            int left = neural.net[i], right = neural.net[i + 1];
            m -= right * left; j -= left; k -= right;

            // hidden gradients - from last hidden to first hidden
            for (int l = 0, w = m; l < left; l++)
            {
                float n = neuron[j + l], sum = 0;
                if (n > 0)
                {
                    for (int r = 0; r < right; r++)
                    {
                        var wr = w + r;
                        var g = gradient[k + r];                       
                        delta[wr] += n * g;
                        sum += neural.weights[wr] * g;
                    }
                    gradient[j + l] = sum;
                }
                w += right;
            }
        }
    }
}
static bool EvalTest(int x, byte label, byte[] samples, Net neural)
{
    Span<float> neuron = new float[neural.neuronLen];
    int p = Eval(x, samples, neural, neuron);
    int t = label;
    return t == p;
}
static int Eval(int x, byte[] samples, Net neural, Span<float> neuron)
{
    FeedInput(x, samples, neuron);
    FeedForward(neural, neuron);
    Softmax(neuron, neural.net[neural.layers]);
    return Argmax(neural, neuron);

    static void FeedInput(int x, byte[] samples, Span<float> neuron)
    {
        for (int i = 0, id = x * 784; i < 784; i++)
        {
            var n = samples[id + i];
            neuron[i] = n > 0 ? n / 255f : 0;
        }
    }
    static void FeedForward(Net neural, Span<float> neuron)
    {
        for (int i = 0, j = 0, k = neural.net[0], m = 0; i < neural.layers; i++)
        {
            int left = neural.net[i], right = neural.net[i + 1];
            for (int l = 0, w = m; l < left; l++)
            {
                float n = neuron[j + l];
                if (n > 0)
                    for (int r = 0; r < right; r++)
                        neuron[k + r] += n * neural.weights[w + r];
                w += right;
            }
            m += left * right; j += left; k += right;
        }
    }
    static void Softmax(Span<float> neuron, int output)
    {
        float scale = 0;
        int N = neuron.Length;
        for (int n = N - output; n < N; n++)
            scale += neuron[n] = MathF.Exp(neuron[n]);
        for (int n = N - output; n < N; n++)
            neuron[n] /= scale;
    }
    static int Argmax(Net neural, Span<float> neuron)
    {
        int output = neural.net[neural.layers];
        int ih = neuron.Length - output;
        float max = neuron[ih];
        int prediction = 0;
        for (int i = 1; i < output; i++)
        {
            float n = neuron[i + ih];
            if (n > max) { max = n; prediction = i; } // grab maxout prediction here
        }
        return prediction;
    }
}
static void Update(float[] weight, float[] delta, float lr, float mom)
{    
     for (int w = 0, W = weight.Length; w < W; w++)
     {
         weight[w] += delta[w] * lr;
         delta[w] *= mom;
     }
}
struct Net
{  
    public readonly int[] net;
    public float[] weights;
    public readonly int neuronLen;
    public readonly int layers;

    public Net(int[] net)
    {
        this.net = net;
        this.weights = Glorot(this.net);
        this.neuronLen = net.Sum(); 
        this.layers = net.Length - 1;
    }
    static float[] Glorot(int[] net)
    {
        int len = GetWeightsLen(net);
        float[] weights = new float[len];
        Erratic rnd = new(12345);

        for (int i = 0, w = 0; i < net.Length - 1; i++, w += net[i - 0] * net[i - 1]) // layer
        {
            float sd = MathF.Sqrt(6.0f / (net[i] + net[i + 1]));
            for (int m = w; m < w + net[i] * net[i + 1]; m++) // weights
                weights[m] = rnd.NextFloat(-sd * 1.0f, sd * 1.0f);
        }
        return weights;
    }
    static int GetWeightsLen(int[] net)
    {
        int sum = 0;
        for (int n = 0; n < net.Length - 1; n++) sum += net[n] * net[n + 1];
        return sum;
    }
    //float learningRate, momentum; 
}
struct AutoData // https://github.com/grensen/easy_regression#autodata
{
    public string source;
    public byte[] samplesTest, labelsTest;
    public byte[] samplesTraining, labelsTraining;
    public AutoData(string yourPath)
    {
        this.source = yourPath;

        // hardcoded urls from my github
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testnLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        // change easy names 
        string d1 = @"trainData", d2 = @"trainLabel", d3 = @"testData", d4 = @"testLabel";

        if (!File.Exists(yourPath + d1)
            || !File.Exists(yourPath + d2)
              || !File.Exists(yourPath + d3)
                || !File.Exists(yourPath + d4))
        {
            System.Console.WriteLine("\nData does not exist");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // padding bits: data = 16, labels = 8
            System.Console.WriteLine("Download MNIST dataset from GitHub");
            this.samplesTraining = (new System.Net.WebClient().DownloadData(trainDataUrl)).Skip(16).Take(60000 * 784).ToArray();
            this.labelsTraining = (new System.Net.WebClient().DownloadData(trainLabelUrl)).Skip(8).Take(60000).ToArray();
            this.samplesTest = (new System.Net.WebClient().DownloadData(testDataUrl)).Skip(16).Take(10000 * 784).ToArray();
            this.labelsTest = (new System.Net.WebClient().DownloadData(testnLabelUrl)).Skip(8).Take(10000).ToArray();

            System.Console.WriteLine("Save cleaned MNIST data into folder " + yourPath + "\n");
            File.WriteAllBytes(yourPath + d1, this.samplesTraining);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, this.samplesTest);
            File.WriteAllBytes(yourPath + d4, this.labelsTest); return;
        }
        // data on the system, just load from yourPath
        System.Console.WriteLine("Load MNIST data and labels from " + yourPath + "\n");
        this.samplesTraining = File.ReadAllBytes(yourPath + d1).Take(60000 * 784).ToArray();
        this.labelsTraining = File.ReadAllBytes(yourPath + d2).Take(60000).ToArray();
        this.samplesTest = File.ReadAllBytes(yourPath + d3).Take(10000 * 784).ToArray();
        this.labelsTest = File.ReadAllBytes(yourPath + d4).Take(10000).ToArray();
    }
}
class Erratic // https://jamesmccaffrey.wordpress.com/2019/05/20/a-pseudo-pseudo-random-number-generator/
{
    private float seed;
    public Erratic(float seed2)
    {
        this.seed = this.seed + 0.5f + seed2;  // avoid 0
    }
    public float Next()
    {
        var x = Math.Sin(this.seed) * 1000;
        var result = (float)(x - Math.Floor(x));  // [0.0,1.0)
        this.seed = result;  // for next call
        return this.seed;
    }
    public float NextFloat(float lo, float hi)
    {
        var x = this.Next();
        return (hi - lo) * x + lo;
    }
};