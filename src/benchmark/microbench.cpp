#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "./flags.h"
#include "./utils.h"

#include "./benchmark_streaming.h"
#include "./concurrent_evaluation.h"

int main(int argc, char **argv) {
    auto flags = parse_flags(argc, argv);
    if (get_boolean_flag(flags, "check_duplicates")) {
        const std::string keys_csv = get_required(flags, "keys_file");
        const std::string keys_file_type = get_with_default(flags, "keys_file_type", "binary");

        std::vector<std::string> files;
        {
            std::istringstream iss(keys_csv);
            std::string v;
            while (std::getline(iss, v, ',')) files.push_back(v);
        }
        if (files.empty()) {
            std::cerr << "[dup_check] no files provided\n";
            return 2;
        }

        int exit_code = 0;
        for (const auto &p : files) {
            auto res = check_duplicates_sorted_scan<uint64_t>(p, keys_file_type, /*length=*/-1);
            if (!res.ok) {
                std::cout << "DupCheck;File=" << p
                          << ";OK=0;"
                          << std::endl;
                exit_code = 1;
                continue;
            }
            std::cout << "DupCheck;File=" << p
                      << ";OK=1"
                      << ";N=" << res.n
                      << ";HasDup=" << (res.has_duplicates ? 1 : 0)
                      << ";DupPairs=" << res.dup_adjacent_pairs;
            if (res.has_duplicates) {
                std::cout << ";FirstDup=" << res.first_dup_value;
                exit_code = 3;
            }
            std::cout << ";" << std::endl;
        }
        return exit_code;
    }
    if (get_boolean_flag(flags, "multithread")) {
        ce::ConcurrentStreamingReplayBenchmark<uint64_t, uint64_t> bench;
        bench.parse_args(argc, argv);
        bench.run();
        return 0;
    }

    bool streaming = get_boolean_flag(flags, "streaming");
    if (streaming) {
        StreamingBenchmark<uint64_t, uint64_t> bench;
        bench.parse_args(argc, argv);
        bench.run();
    } else {
        std::cerr << "[micro] error: non-streaming benchmark is not available "
                     "(benchmark.h has been removed). Use --streaming or --multithread, "
                     "or run the duplicate checker with --check_duplicates.\n";
        return 1;
    }
}