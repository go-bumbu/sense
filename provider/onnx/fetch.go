package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/go-bumbu/sense/internal/fetch"
)

// Pinned runtime. The onnxruntime_go bindings are written against one
// release's C API headers, so the build downloaded here is bumped together
// with the binding version in go.mod; pins_test.go fails until pinnedBinding
// matches the module version again. The builds come from go-bumbu/sense-deps,
// which repackages Microsoft's signed NuGet builds in the layout of the
// upstream release tarballs (the upstream GitHub release has been unreachable
// since 2026-11-17); the digests are the ones its assets.json pins.
const (
	// pinnedBinding is the onnxruntime_go release whose headers LibVersion
	// matches.
	pinnedBinding = "v1.38.0"
	// LibVersion is the ONNX Runtime release a managed LibDir downloads;
	// name a managed directory after it so a bump never loads stale native
	// code. runtimeLoader.load refuses a library of another minor version.
	LibVersion = "1.31.1"
	// libMajorMinor is what the loaded library's version must start with.
	libMajorMinor = "1.31"
	// installRecordFile records which build a managed directory holds, so a
	// change of Provider refills it; an operator directory with one is
	// checked against it.
	installRecordFile = "onnxruntime-install.json"
	// maxDownloadBytes caps any single download; a download pinned to a
	// larger exact size (the CUDA build) is capped at that size instead.
	maxDownloadBytes = 1 << 30
	// maxLibraryBytes caps one extracted library; the CUDA provider's is
	// the largest by far.
	maxLibraryBytes = 2 << 30
	// downloadTimeout bounds the first-use downloads as a whole: the model
	// is ~370 MB and the CUDA build ~245 MB.
	downloadTimeout = 30 * time.Minute
	// userAgent names this package's downloads.
	userAgent = "sense-onnx"
)

// Variants of the runtime build; the key of libAssets is platform/variant.
const (
	variantCPU  = "cpu"
	variantCUDA = "cuda"
)

// asset is one downloadable file with its expected digest.
type asset = fetch.Asset

// depsURL is the release download base of go-bumbu/sense-deps.
const depsURL = fetch.DepsURL

// libAssets are the release builds of LibVersion per GOOS/GOARCH and
// variant. The CPU builds cover the three platforms the release ships as
// tarballs (the macOS one includes CoreML); the CUDA build is the CUDA 13
// one for linux/amd64. Other platforms and builds (Windows, CUDA 12,
// TensorRT, ROCm) go into an operator-supplied LibDir.
var libAssets = map[string]asset{
	"linux/amd64/" + variantCPU: {
		URL:    depsURL + "onnxruntime-v1.31.1/onnxruntime-linux-x64-1.31.1.tgz",
		SHA256: "67ffe9df0e5bae1bd8f3215768c464db536b44b6fe85e0e6bb6a1fce498fe624",
		Size:   11413374,
	},
	"linux/arm64/" + variantCPU: {
		URL:    depsURL + "onnxruntime-v1.31.1/onnxruntime-linux-aarch64-1.31.1.tgz",
		SHA256: "6f4b8bd6199aa6c51d5bd493cfcd96b33bc8f552f6d2afc8aac337765d15d174",
		Size:   10453595,
	},
	"darwin/arm64/" + variantCPU: {
		URL:    depsURL + "onnxruntime-v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz",
		SHA256: "2a5ff4b116085093b19a371b953c27a53ac6cb0ebbe397bc6eade860df7ff98c",
		Size:   12948953,
	},
	"linux/amd64/" + variantCUDA: {
		URL:    depsURL + "onnxruntime-v1.31.1/onnxruntime-linux-x64-gpu_cuda13-1.31.1.tgz",
		SHA256: "46aec7af4f3143d4f33bbbf449c77e8da8d5073d2d4ef435550af82dcea517eb",
		Size:   244731742,
	},
}

// httpClient is the client downloads go through. The default client honours
// HTTP(S)_PROXY; tests point assets at an httptest server instead.
var httpClient = http.DefaultClient

// platform is the first part of the libAssets key for this binary.
func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

// variantFor is the runtime build a provider needs: CUDA has its own, the
// others are in the default build of their platform.
func variantFor(p Provider) string {
	if p == ProviderCUDA {
		return variantCUDA
	}
	return variantCPU
}

// libFile is the versioned library file of LibVersion, the one the
// bindings are pointed at.
func libFile() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime." + LibVersion + ".dylib"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so." + LibVersion
}

// plainLibFile is the unversioned name, what a manual install may be left
// with; the version check after load tells whether it is the right one.
func plainLibFile() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so"
}

// libraryPath is the library to load from libDir: the versioned file, else
// the plain one.
func libraryPath(libDir string) (string, error) {
	for _, name := range []string{libFile(), plainLibFile()} {
		p := filepath.Join(libDir, name)
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no %s in %s", libFile(), libDir)
}

// installRecord is what a managed install writes next to the libraries.
type installRecord struct {
	Version string `json:"version"`
	Variant string `json:"variant"`
}

// readInstallRecord returns the directory's record, nil when it has none.
func readInstallRecord(libDir string) (*installRecord, error) {
	data, err := os.ReadFile(filepath.Join(libDir, installRecordFile)) //nolint:gosec // libDir is operator configuration and installRecordFile a fixed name; the content is only parsed as JSON
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var rec installRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("parse %s in %s: %w", installRecordFile, libDir, err)
	}
	return &rec, nil
}

// downloader is how this package's downloads are made (see fetch.Downloader):
// through httpClient, capped at maxDownloadBytes, each logged to logger under
// this package's component.
func downloader(logger *slog.Logger) fetch.Downloader {
	return fetch.Downloader{Client: httpClient, UserAgent: userAgent, Limit: maxDownloadBytes, Logger: logger.With(logComponent)}
}

// downloadContext bounds the first-use downloads: by downloadTimeout when
// the caller's ctx has no deadline, and by the caller's own otherwise, which
// a default must not cut short (see fetch.WithDefaultTimeout).
func downloadContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return fetch.WithDefaultTimeout(ctx, downloadTimeout)
}

// ensureModelFile makes sure spec.File exists, downloading spec.URL into it
// (verified against spec.SHA256 and, when set, spec.Size) when it is absent;
// a download is logged to logger.
func ensureModelFile(ctx context.Context, spec ModelSpec, logger *slog.Logger) error {
	a := asset{URL: spec.URL, SHA256: spec.SHA256, Size: spec.Size}
	return downloader(logger).EnsureFile(ctx, "model "+spec.Name, a, spec.File)
}

// libsWant is what ensureLibs is asked for: a directory, the runtime build
// it should hold, and what to do with it once it does.
type libsWant struct {
	// plat is the first part of the libAssets key, "linux/amd64".
	plat string
	// dir is the absolute library directory.
	dir string
	// managed marks dir as owned by this package (Runtime.ManagedLibDir).
	managed bool
	// variant is the build dir should hold (see variantFor).
	variant string
	// loadedDir is the directory this process loaded the runtime from, ""
	// while it has not; nil means it has not.
	loadedDir func() string
	// use loads the library from dir. It runs while a managed dir is locked,
	// so no other process replaces it halfway; nil does nothing.
	use func() error
}

// ensureLibs makes sure w.dir holds the ONNX Runtime build of w.variant and
// then runs w.use on it. A managed directory is filled from libAssets when
// it is empty or when its record names another version or variant, under
// the directory's lock (see fetch.InstallDir), and the download and install
// are logged to logger. It is never filled over files this package did not
// install, and never while this process runs the library in it; a process
// that does run it, and wants nothing else, does not wait for the lock. An
// operator directory is only checked, never written to.
func ensureLibs(ctx context.Context, w libsWant, logger *slog.Logger) error {
	if !w.managed {
		if _, err := libraryPath(w.dir); err != nil {
			return fmt.Errorf("no %s in %s: unpack the lib/ directory of the onnxruntime %s release there (the %s build for %s), or use a managed LibDir, which gets it downloaded", libFile(), w.dir, LibVersion, w.variant, w.plat)
		}
		rec, err := readInstallRecord(w.dir)
		if err != nil {
			return err
		}
		if err := checkInstallRecord(w.dir, rec, w.variant); err != nil {
			return err
		}
		return w.load()
	}
	record, err := json.Marshal(installRecord{Version: LibVersion, Variant: w.variant})
	if err != nil {
		return err
	}
	installed, err := downloader(logger).InstallDir(ctx, fetch.DirInstall{
		Dir: w.dir, What: "onnxruntime " + LibVersion + " (" + w.variant + ")", Archive: libAssets[w.plat+"/"+w.variant], Extract: extractLibs,
		RecordFile: installRecordFile, Record: record,
		Ready: func() (bool, error) { return w.ready(logger) },
		Use:   w.load,
		// a process that runs the library from here does not queue behind
		// another process's download at its next open
		Loaded: func() bool { return w.loadedDir != nil && w.loadedDir() == w.dir },
	})
	if installed {
		logger.Info("installed onnxruntime "+LibVersion, logComponent, slog.String("dir", w.dir), slog.String("variant", w.variant))
	}
	return err
}

// load runs w.use, which may be nil.
func (w libsWant) load() error {
	if w.use == nil {
		return nil
	}
	return w.use()
}

// ready reports whether the managed w.dir can be used as it is, and
// refuses, with an error, an install that cannot be made: of a build that
// is not pinned for the platform, or over a library this process has
// already loaded, which cannot be swapped under it. A loaded CUDA build
// stays for a CPU request: it runs the CPU provider too.
func (w libsWant) ready(logger *slog.Logger) (bool, error) {
	rec, err := readInstallRecord(w.dir)
	if err != nil {
		return false, err
	}
	_, missing := libraryPath(w.dir)
	present := missing == nil
	if present && rec != nil && rec.Version == LibVersion && rec.Variant == w.variant {
		return true, nil
	}
	if w.loadedDir != nil && w.loadedDir() == w.dir {
		if rec != nil && rec.Variant == variantCUDA && w.variant == variantCPU {
			return true, nil
		}
		return false, fmt.Errorf("onnxruntime is already loaded from %s, which does not hold the %s build: a library a process has loaded cannot be replaced under it; open with that Provider first, or restart", w.dir, w.variant)
	}
	if _, ok := libAssets[w.plat+"/"+w.variant]; !ok {
		return false, fmt.Errorf("no pinned onnxruntime %s build for %s: use an unmanaged LibDir holding the release's lib/ directory", w.variant, w.plat)
	}
	if present {
		logger.Info("replacing the onnxruntime install", logComponent, slog.String("dir", w.dir), slog.Any("record", rec), slog.String("want", LibVersion+"/"+w.variant))
	}
	return false, nil
}

// checkInstallRecord refuses an operator directory whose record names
// another version or variant than the one asked for. A directory without a
// record (manual install) passes here: the version is checked after load,
// and a build without the provider fails when the session is created.
func checkInstallRecord(libDir string, rec *installRecord, variant string) error {
	if rec == nil || (rec.Version == LibVersion && rec.Variant == variant) {
		return nil
	}
	return fmt.Errorf("%s holds onnxruntime %s (%s) but %s (%s) is wanted: install that build there, or use a managed LibDir", libDir, rec.Version, rec.Variant, LibVersion, variant)
}

// wantLib reports whether an archive entry is one of the runtime libraries:
// a libonnxruntime*.so* or *.dylib directly under lib/ (the core library,
// its symlinks, and the provider libraries of the CUDA build). The headers,
// pkgconfig and cmake files, and the macOS debug symbols, which carry the
// library's own name inside a .dSYM directory, are skipped.
func wantLib(p string) bool {
	if path.Base(path.Dir(p)) != "lib" {
		return false
	}
	name := path.Base(p)
	return strings.HasPrefix(name, "libonnxruntime") && (strings.Contains(name, ".so") || strings.Contains(name, ".dylib"))
}

// extractLibs copies the runtime libraries out of a release tarball into
// dir, flattened (see fetch.ExtractTarGz), and checks the versioned library
// is among them.
func extractLibs(archive, dir string) error {
	if _, err := fetch.ExtractTarGz(archive, dir, wantLib, maxLibraryBytes); err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(dir, libFile())); err != nil {
		return fmt.Errorf("no %s in %s", libFile(), archive)
	}
	return nil
}
