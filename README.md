# Multi-Core Neural Network Demo Using C#


## Intuition
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_intuition.png?raw=true">
</p>

## First Start
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_init.png?raw=true">
</p>

## Batchsize 800
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_800.png?raw=true">
</p>

## Single vs. Multi Core Batch Training Code Details
~~~cs
for (int b = 0; b < B; b++)
    if (multiCore)
    {
        System.Threading.Tasks.Parallel.ForEach(
            System.Collections.Concurrent.Partitioner.Create(b * BATCHSIZE, (b + 1) * BATCHSIZE), range =>
            {
                for (int x = range.Item1, X = range.Item2; x < X; x++)
                    c[x] = EvalAndTrain(x, d.samplesTraining, neural, delta, d.labelsTraining[x]);
            });
        System.Threading.Tasks.Parallel.ForEach(
            System.Collections.Concurrent.Partitioner.Create(0, neural.weights.Length), range =>
            {
                UpdateWeights(range.Item1, range.Item2, neural.weights, delta, lr, mom);
            });
    }
    else
    {
        for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
            c[x] = EvalAndTrain(x, d.samplesTraining, neural, delta, d.labelsTraining[x]);
        UpdateWeights(0, neural.weights.Length, neural.weights, delta, lr, mom);
    }
~~~

## Batchsize 800 With .NET 7
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_800_dotnet7.png?raw=true">
</p>

## Batchsize 400
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_400.png?raw=true">
</p>

## Batchsize 200
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_200.png?raw=true">
</p>

## Batchsize 100
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_100.png?raw=true">
</p>

## Batchsize 50
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_50.png?raw=true">
</p>

## Batchsize 10
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_10.png?raw=true">
</p>

## Batchsize 1
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/multi-core_batch_1.png?raw=true">
</p>

## Floating Point Issues
~~~cs
float[] array = new float[10];
for (int i = 0; i < array.Length; i++)
{
    array[i] = 0.01f * (i + 1);
}
float a = 0, b = 0;
for (int i = 0; i < 10; i++)
{
    a += array[i];
    b += array[9 - i];
}
System.Console.WriteLine("a = " + a + " b = " + b);
// output: a = 0.55 b = 0.54999995
~~~

## Test It On [.NET Fiddle](https://dotnetfiddle.net/)
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/dotnetfiddle_floating_point_issue.png?raw=true">
</p>

## Code Intuition
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/network_intuition.png?raw=true">
</p>

https://browser.geekbench.com/processor-benchmarks
## 10 Years CPU Benchmark Single-Core 
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/bench_decade_single_core.gif?raw=true">
</p>

## 10 Years CPU Benchmark Multi-Core 
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/bench_decade_multi_core.gif?raw=true">
</p>

## The Used CPU - AMD 5600x 
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/used_cpu_amd_5600x.png?raw=true">
</p>

## A part of my CPU history
<p align="center">
  <img src="https://github.com/grensen/multi-core/blob/main/figures/cpu_generations.jpg_rdy.png?raw=true">
</p>
