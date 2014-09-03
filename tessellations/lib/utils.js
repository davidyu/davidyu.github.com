var Utils = (function () {
    function Utils() {
    }
    Utils.deepCopy = function (src) {
        var dst = {};
        var keys = Object.keys(src);
        keys.map(function (key) {
            return dst[key] = src[key];
        });
        return dst;
    };
    return Utils;
})();
