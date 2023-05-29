#include <iostream>
#include <omp.h>

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr) / sizeof(arr[0]);

    int min_val = arr[0];
    int max_val = arr[0];
    int sum = 0;

    #pragma omp parallel for reduction(min: min_val) reduction(max: max_val) reduction(+: sum)
    for (int i = 0; i < n; ++i) {
        if (arr[i] < min_val)
            min_val = arr[i];

        if (arr[i] > max_val)
            max_val = arr[i];

        sum += arr[i];
    }

    double average = static_cast<double>(sum) / n;

    std::cout << "Minimum value: " << min_val << std::endl;
    std::cout << "Maximum value: " << max_val << std::endl;
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Average: " << average << std::endl;

    return 0;
}