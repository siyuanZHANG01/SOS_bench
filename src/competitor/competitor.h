#pragma once

#include"./indexInterface.h"
#include "./alex/alex.h"
#include "./alexol/alex.h"
// alexol defines several very generic macros in its headers; avoid leaking them
// to other competitors.
#ifdef CAS
#undef CAS
#endif
#ifdef LOG
#undef LOG
#endif
#ifdef LOG_FATAL
#undef LOG_FATAL
#endif
#ifdef PRINT_DEBUG
#undef PRINT_DEBUG
#endif

#include "./artsync/artrowex.h"
#include "./artsync/artolc.h"
#include "./artsync/artunsync.h"
#include "./xindex/xindex.h"
#include "./btreeolc/btreeolc.h"
#include "./hot/hot.h"
#include "./hot/hotrowex.h"
#include "./lipp/lipp.h"
#ifdef RT_ASSERT
#undef RT_ASSERT
#endif
#ifdef PRINT_DEBUG
#undef PRINT_DEBUG
#endif
#include "./lippol/lippol.h"
#ifdef RT_ASSERT
#undef RT_ASSERT
#endif
#ifdef PRINT_DEBUG
#undef PRINT_DEBUG
#endif
#include "pgm/pgm.h"
#include "btree/btree.h"
// #include "wormhole/wormhole.h"
// Prevent wormhole_u64 from overriding the benchmark's likely/unlikely macros
// (defined in src/benchmark/utils.h).
#ifdef likely
#undef likely
#endif
#ifdef unlikely
#undef unlikely
#endif
#include "wormhole_u64/wormhole_u64.h"
#ifdef likely
#undef likely
#endif
#ifdef unlikely
#undef unlikely
#endif
// Restore the benchmark versions to keep behavior consistent.
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#include "masstree/masstree.h"
// masstree leaks a very generic macro `K` (e.g., `#define K 2`) which can break
// other competitors that legitimately use `K` as a template parameter name.
// Keep it scoped to masstree's headers.
#ifdef K
#undef K
#endif
// finedex also defines likely/unlikely and LOG; keep them scoped.
#ifdef likely
#undef likely
#endif
#ifdef unlikely
#undef unlikely
#endif
#ifdef LOG
#undef LOG
#endif
#include "finedex/finedex.h"
#ifdef likely
#undef likely
#endif
#ifdef unlikely
#undef unlikely
#endif
#ifdef LOG
#undef LOG
#endif
// Restore the benchmark versions again (in case finedex overwrote them).
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#include "./carmi/carmi.h"
// LOFT has a few global identifiers that can conflict with other bundled
// components (e.g., Masstree's global `result_t`, and GRE's helper functions
// `get_search_keys*` in src/benchmark/utils.h). Keep LOFT's globals isolated by
// renaming them for the duration of the LOFT include.
#define result_t loft_result_t
#define get_search_keys loft_get_search_keys
#define get_search_keys_zipf loft_get_search_keys_zipf
// Also isolate common helper names and zipf generator types that otherwise
// collide with GRE's own versions.
#define memory_fence loft_memory_fence
#define fence loft_fence
#define cmpxchg loft_cmpxchg
#define cmpxchgb loft_cmpxchgb
#define ScrambledZipfianGenerator loft_ScrambledZipfianGenerator
#include "./loft/loft.h"
#undef ScrambledZipfianGenerator
#undef cmpxchgb
#undef cmpxchg
#undef fence
#undef memory_fence
#undef get_search_keys_zipf
#undef get_search_keys
#undef result_t
// LOFT also defines CAS; prevent it from affecting other headers included later.
#ifdef CAS
#undef CAS
#endif
#ifdef LOG
#undef LOG
#endif

#include "./dili/dili.h"
#include "./sali/sali.h"
#ifdef RT_ASSERT
#undef RT_ASSERT
#endif
#ifdef PRINT_DEBUG
#undef PRINT_DEBUG
#endif
#include "./dytis/dytis.h"

// Some third-party competitors (e.g., CARMI) define very generic debug macros
// like `DEBUG` in their headers. That leaks into other competitors and can
// accidentally enable debug-only includes (e.g., SWIX expects debug.hpp which
// isn't shipped here). Keep the benchmark harness build stable by preventing
// such macro pollution from affecting other adapters.
#ifdef DEBUG
#undef DEBUG
#endif
#ifdef DEBUG_KEY
#undef DEBUG_KEY
#endif

#include "./imtree/imtree.h"
#include "./swix/swix.h"
// Parallel SWIX (PSWIX) adapter (used by concurrent replay benchmark).
#include "./swix/pswix.h"
#include "iostream"

template<class KEY_TYPE, class PAYLOAD_TYPE>
indexInterface<KEY_TYPE, PAYLOAD_TYPE> *get_index(std::string index_type) {
  indexInterface<KEY_TYPE, PAYLOAD_TYPE> *index;
  if (index_type == "alexol") {
    index = new alexolInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if(index_type == "alex") {
    index = new alexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "btreeolc") {
    index = new BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  // else if (index_type == "wormhole") {
  //   index = new WormholeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  // }
  else if (index_type == "wormhole_u64") {
    index = new WormholeU64Interface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if( index_type == "hot") {
    index = new HotInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if( index_type == "hotrowex") {
    index = new HotRowexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "masstree") {
    index = new MasstreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "xindex") {
    index = new xindexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "pgm") {
    index = new pgmInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if(index_type == "btree") {
    index = new BTreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "artolc") {
    index = new ARTOLCInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  // else if (index_type == "artrowex") {
  //   index = new ARTROWEXInterface<KEY_TYPE, PAYLOAD_TYPE>;
  // }
  else if (index_type == "artunsync") {
    index = new ARTUnsynchronizedInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "lippol") {
    index = new LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "lipp") {
    index = new LIPPInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "finedex") {
    index = new finedexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "dili") {
    index = new diliInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "sali") {
    index = new saliInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "dytis") {
    index = new dytisInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "loft") {
    index = new loftInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "carmi") {
    index = new carmiInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "swix") {
    index = new swixInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "imtree") {
    index = new imtreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else if (index_type == "pswix") {
    index = new pswixInterface<KEY_TYPE, PAYLOAD_TYPE>;
  }
  else {
    std::cout << "Could not find a matching index called " << index_type << ".\n";
    exit(0);
  }

  return index;
}