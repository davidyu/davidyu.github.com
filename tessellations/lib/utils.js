var Utils = (function () {
    function Utils() {
    }
    Utils.deepCopy = function (src) {
        var dst = Object.create(Object.getPrototypeOf(src));
        var keys = Object.keys(src);
        keys.map(function (key) {
            return dst[key] = src[key];
        });
        return dst;
    };

    // http://stackoverflow.com/a/11582513
    Utils.getURLParameter = function (key) {
        return decodeURIComponent((new RegExp('[?|&]' + key + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search) || [, ""])[1].replace(/\+/g, '%20')) || null;
    };

    Utils.getOffset = function (elem) {
        var left = 0;
        var top = 0;
        while (true) {
            left += elem.offsetLeft;
            top += elem.offsetTop;
            if (elem.offsetParent === null) {
                break;
            }
            elem = elem.offsetParent;
        }
        return { top: top, left: left };
    };

    Utils.EPSILON = 0.00001;
    return Utils;
})();
