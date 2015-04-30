var MIN_VAL = 2;
var DrawBackend;
(function (DrawBackend) {
    DrawBackend[DrawBackend["CANVAS"] = 0] = "CANVAS";
    DrawBackend[DrawBackend["SVG"] = 1] = "SVG";
})(DrawBackend || (DrawBackend = {}));
var GameType;
(function (GameType) {
    GameType[GameType["SURVIVAL"] = 0] = "SURVIVAL";
    GameType[GameType["PUZZLE"] = 1] = "PUZZLE";
})(GameType || (GameType = {}));
var Utils = (function () {
    function Utils() {
    }
    Utils.deepCopy = function (src) {
        var dst = Object.create(Object.getPrototypeOf(src));
        var keys = Object.keys(src);
        keys.map(function (key) { return dst[key] = src[key]; });
        return dst;
    };
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
        return [left, top];
    };
    Utils.shuffle = function (deck) {
        var counter = deck.length, temp, index;
        while (counter > 0) {
            index = (Math.random() * counter--) | 0;
            temp = deck[counter];
            deck[counter] = deck[index];
            deck[index] = temp;
        }
    };
    Utils.getViewport = function () {
        var viewPortWidth;
        var viewPortHeight;
        if (typeof window.innerWidth != 'undefined') {
            viewPortWidth = window.innerWidth, viewPortHeight = window.innerHeight;
        }
        else if (typeof document.documentElement != 'undefined' && typeof document.documentElement.clientWidth != 'undefined' && document.documentElement.clientWidth != 0) {
            viewPortWidth = document.documentElement.clientWidth, viewPortHeight = document.documentElement.clientHeight;
        }
        else {
            viewPortWidth = document.getElementsByTagName('body')[0].clientWidth, viewPortHeight = document.getElementsByTagName('body')[0].clientHeight;
        }
        return [viewPortWidth, viewPortHeight];
    };
    Utils.getScrollOffset = function () {
        var top = window.pageYOffset || document.documentElement.scrollTop, left = window.pageXOffset || document.documentElement.scrollLeft;
        return [left, top];
    };
    Utils.truncate = function (x) {
        return Math[x < 0 ? 'ceil' : 'floor'](x);
    };
    Utils.EPSILON = 0.00001;
    return Utils;
})();
var TileType;
(function (TileType) {
    TileType[TileType["OUT_OF_BOUNDS"] = 0] = "OUT_OF_BOUNDS";
    TileType[TileType["EMPTY"] = 1] = "EMPTY";
    TileType[TileType["REGULAR"] = 2] = "REGULAR";
    TileType[TileType["CONCRETE"] = 3] = "CONCRETE";
    TileType[TileType["LAVA"] = 4] = "LAVA";
})(TileType || (TileType = {}));
;
var CardinalDirection;
(function (CardinalDirection) {
    CardinalDirection[CardinalDirection["NORTH"] = 0] = "NORTH";
    CardinalDirection[CardinalDirection["EAST"] = 1] = "EAST";
    CardinalDirection[CardinalDirection["SOUTH"] = 2] = "SOUTH";
    CardinalDirection[CardinalDirection["WEST"] = 3] = "WEST";
})(CardinalDirection || (CardinalDirection = {}));
var Dim2;
(function (Dim2) {
    Dim2[Dim2["X"] = 0] = "X";
    Dim2[Dim2["Y"] = 1] = "Y";
})(Dim2 || (Dim2 = {}));
var CartesianCoords = (function () {
    function CartesianCoords(x, y) {
        this.x = x;
        this.y = y;
    }
    CartesianCoords.prototype.equals = function (that) {
        return that != null && this.x == that.x && this.y == that.y;
    };
    CartesianCoords.prototype.displace = function (arg, magnitude) {
        if (magnitude === void 0) { magnitude = 1; }
        if (arg instanceof CartesianCoords) {
            var delta = arg;
            return new CartesianCoords(this.x + delta.x, this.y + delta.y);
        }
        else {
            switch (arg) {
                case 0 /* NORTH */:
                    return new CartesianCoords(this.x, this.y - magnitude);
                    break;
                case 2 /* SOUTH */:
                    return new CartesianCoords(this.x, this.y + magnitude);
                    break;
                case 3 /* WEST */:
                    return new CartesianCoords(this.x - magnitude, this.y);
                    break;
                case 1 /* EAST */:
                    return new CartesianCoords(this.x + magnitude, this.y);
                    break;
            }
        }
    };
    CartesianCoords.prototype.displaceAndWrap = function (arg, magnitude, gridw, gridh) {
        if (arg instanceof CartesianCoords) {
            var delta = arg;
            if (delta.x < 0) {
                delta.x += gridw;
            }
            if (delta.y < 0) {
                delta.y += gridh;
            }
            return new CartesianCoords(this.x + delta.x, this.y + delta.y);
        }
        else {
            switch (arg) {
                case 0 /* NORTH */:
                    return new CartesianCoords(this.x, (this.y + gridh - magnitude) % gridh);
                    break;
                case 2 /* SOUTH */:
                    return new CartesianCoords(this.x, (this.y + magnitude) % gridh);
                    break;
                case 3 /* WEST */:
                    return new CartesianCoords((this.x + gridw - magnitude) % gridw, this.y);
                    break;
                case 1 /* EAST */:
                    return new CartesianCoords((this.x + magnitude) % gridw, this.y);
                    break;
            }
        }
    };
    CartesianCoords.displace = function (coords, direction) {
        return new CartesianCoords(coords.x + direction.x, coords.y + direction.y);
    };
    CartesianCoords.displaceAndWrap = function (coords, direction, steps, gridw, gridh) {
        if (direction.x < 0) {
            direction.x += gridw;
        }
        if (direction.y < 0) {
            direction.y += gridh;
        }
        return new CartesianCoords((coords.x + direction.x * steps) % gridw, (coords.y + direction.y * steps) % gridh);
    };
    CartesianCoords.prototype.toString = function () {
        return "( " + this.x + ", " + this.y + " )";
    };
    return CartesianCoords;
})();
var CartesianBounds = (function () {
    function CartesianBounds(n, e, s, w) {
        this.n = n != undefined ? n : -1;
        this.e = e != undefined ? e : -1;
        this.s = s != undefined ? s : -1;
        this.w = w != undefined ? s : -1;
    }
    return CartesianBounds;
})();
var Tile = (function () {
    function Tile(t, v) {
        this.type = t;
        this.value = v;
        this.bounds = new CartesianBounds();
    }
    Tile.prototype.toString = function () {
        return this.value.toString();
    };
    Tile.prototype.isTangible = function () {
        return this.type != 1 /* EMPTY */;
    };
    return Tile;
})();
var Model;
(function (Model) {
    var Square = (function () {
        function Square(gridw, gridh, maxgridw, maxgridh, DefaultTile, OutOfBoundsTile) {
            this.outOfBoundsTile = OutOfBoundsTile;
            this.gridw = gridw;
            this.gridh = gridh;
            this.maxgridw = maxgridw;
            this.maxgridh = maxgridh;
            this.size = gridw * gridh;
            this.grid = [];
            for (var i = 0; i < this.gridw * this.gridh; i++) {
                this.grid.push(DefaultTile);
            }
            this.modelSignals = {
                moved: new signals.Signal(),
                crushed: new signals.Signal(),
                deleted: new signals.Signal()
            };
        }
        Square.prototype.toCoords = function (i) {
            return new CartesianCoords(i % this.gridw, Math.floor(i / this.gridw));
        };
        Square.prototype.toFlippedCoords = function (c) {
            return new CartesianCoords(this.gridw - 1 - c.x, this.gridh - 1 - c.y);
        };
        Square.prototype.toFlippedDirection = function (d) {
            switch (d) {
                case 0 /* NORTH */: return 2 /* SOUTH */;
                case 2 /* SOUTH */: return 0 /* NORTH */;
                case 1 /* EAST */: return 3 /* WEST */;
                case 3 /* WEST */: return 1 /* EAST */;
            }
        };
        Square.prototype.import = function (lyt, flip) {
            if (flip === void 0) { flip = false; }
            var tiles = null;
            {
                tiles = lyt.map(function (n) {
                    if (n != 0) {
                        return new Tile(2 /* REGULAR */, n);
                    }
                    else {
                        return new Tile(1 /* EMPTY */, -1);
                    }
                });
                if (flip)
                    tiles.reverse();
            }
            this.grid = tiles;
            this.recomputeAllBounds();
        };
        Square.prototype.toFlat = function (x, y) {
            return x + y * this.gridw;
        };
        Square.prototype.get = function (c) {
            return c.y >= 0 && c.x >= 0 && c.x < this.gridw && c.y < this.gridh ? this.grid[c.x + c.y * this.gridw] : this.outOfBoundsTile;
        };
        Square.prototype.getFlat = function (i) {
            return i >= 0 && i < this.gridw * this.gridh ? this.grid[i] : this.outOfBoundsTile;
        };
        Square.prototype.set = function (c, tile) {
            if (c.y >= 0 && c.x >= 0 && c.x < this.gridw && c.y < this.gridh) {
                this.grid[c.x + c.y * this.gridw] = tile;
            }
        };
        Square.prototype.setFlat = function (i, tile) {
            if (i >= 0 && i < this.gridw * this.gridh) {
                this.grid[i] = tile;
            }
        };
        Square.prototype.isEmpty = function (arg) {
            if (typeof (arg) == "number") {
                return this.getFlat(arg).type == 1 /* EMPTY */;
            }
            else {
                return this.get(arg).type == 1 /* EMPTY */;
            }
        };
        Square.prototype.getTileArray = function () {
            return this.grid;
        };
        Square.prototype.getCol = function (arg) {
            var col = [];
            if (typeof (arg) == "number") {
                var x = arg;
                for (var y = 0; y < this.gridw; y++) {
                    col.push(new CartesianCoords(x, y));
                }
            }
            else {
                var x = arg.x;
                for (var y = 0; y < this.gridw; y++) {
                    col.push(new CartesianCoords(x, y));
                }
            }
            return col;
        };
        Square.prototype.getRow = function (arg) {
            var row = [];
            if (typeof (arg) == "number") {
                var y = arg;
                for (var x = 0; x < this.gridw; x++) {
                    row.push(new CartesianCoords(x, y));
                }
            }
            else {
                var y = arg.y;
                for (var x = 0; x < this.gridw; x++) {
                    row.push(new CartesianCoords(x, y));
                }
            }
            return row;
        };
        Square.prototype.floodAcquire = function (start) {
            var cluster = [];
            var marked = { get: null, set: null };
            marked.get = function (key) {
                return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
            };
            marked.set = function (key) {
                this[JSON.stringify(key)] = true;
            };
            var Q = [];
            var tile = this.get(start);
            if (tile == this.outOfBoundsTile) {
                return [];
            }
            if (this.get(new CartesianCoords(start.x, start.y)) != tile)
                return [];
            Q.push(start);
            while (Q.length > 0) {
                var n = Q.shift();
                if (this.get(n).value == tile.value && this.get(n).type == tile.type && !marked.get(n)) {
                    var w = new CartesianCoords(n.x, n.y);
                    var e = new CartesianCoords(n.x, n.y);
                    while (this.get(new CartesianCoords(w.x - 1, w.y)).value == tile.value && this.get(new CartesianCoords(w.x - 1, w.y)).type == tile.type) {
                        w.x--;
                    }
                    while (this.get(new CartesianCoords(e.x + 1, e.y)).value == tile.value && this.get(new CartesianCoords(e.x + 1, e.y)).type == tile.type) {
                        e.x++;
                    }
                    for (var x = w.x; x < e.x + 1; x++) {
                        var nn = new CartesianCoords(x, n.y);
                        marked.set(nn);
                        cluster.push(nn);
                        var north = new CartesianCoords(nn.x, nn.y - 1);
                        var south = new CartesianCoords(nn.x, nn.y + 1);
                        if (this.get(north).value == tile.value && this.get(north).type == tile.type)
                            Q.push(north);
                        if (this.get(south).value == tile.value && this.get(south).type == tile.type)
                            Q.push(south);
                    }
                }
            }
            return cluster;
        };
        Square.prototype.move = function (group, direction, step) {
            var _this = this;
            var to = group.map(function (c) {
                return c.displaceAndWrap(direction, step, _this.gridw, _this.gridh);
            });
            var cachedGroupTiles = group.map(function (c) {
                return _this.get(c);
            });
            to.forEach(function (t, i) {
                _this.set(t, new Tile(cachedGroupTiles[i].type, cachedGroupTiles[i].value));
            });
        };
        Square.prototype.prune = function (target) {
            var _this = this;
            console.log("may prune at " + target);
            var startTile = this.get(target);
            if (!startTile.isTangible())
                return;
            var targets = this.floodAcquire(target);
            if (targets.length >= startTile.value) {
                var str = "pruning: ";
                targets.forEach(function (t) {
                    str += _this.get(t).value + " ";
                });
                console.log(str);
                this.modelSignals.deleted.dispatch(targets);
                targets.forEach(function (t, i) {
                    _this.set(t, new Tile(1 /* EMPTY */, -1));
                });
                console.log("after prune:");
                console.log(this.debugPrint());
            }
        };
        Square.prototype.checkCollision = function (from, future) {
            var _this = this;
            return future.map(function (cell, i) {
                var ignoreCollision = from.some(function (c) {
                    return c != undefined && c.x == cell.x && c.y == cell.y;
                });
                var cellIsOutofBounds = _this.get(cell).type == 0 /* OUT_OF_BOUNDS */;
                var isCollision = cellIsOutofBounds || (!ignoreCollision && _this.get(cell).isTangible());
                return isCollision;
            });
        };
        Square.prototype.recomputeAllBounds = function () {
            var _this = this;
            this.grid.forEach(function (t, i) {
                if (t.isTangible()) {
                    t.bounds = _this.computeBounds(_this.toCoords(i));
                }
            });
        };
        Square.prototype.getTileBoundInDirection = function (c, dir) {
            var b = this.get(c).bounds;
            switch (dir) {
                case 0 /* NORTH */: return new CartesianCoords(c.x, b.n);
                case 2 /* SOUTH */: return new CartesianCoords(c.x, b.s);
                case 3 /* WEST */: return new CartesianCoords(b.w, c.y);
                case 1 /* EAST */: return new CartesianCoords(b.e, c.y);
            }
        };
        Square.prototype.computeBounds = function (c) {
            var group = this.floodAcquire(c);
            var i = -1;
            for (var i = 0; i < group.length; i++) {
                if (JSON.stringify(group[i]) == JSON.stringify(c)) {
                    break;
                }
            }
            var dest;
            var future;
            function resetDestFuture() {
                dest = group.map(Utils.deepCopy);
                future = dest.map(Utils.deepCopy);
            }
            resetDestFuture();
            var bounds = new CartesianBounds();
            var n = new CartesianCoords(0, -1);
            var s = new CartesianCoords(0, 1);
            var w = new CartesianCoords(-1, 0);
            var e = new CartesianCoords(1, 0);
            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(n);
                });
            }
            bounds.n = dest[i].y;
            resetDestFuture();
            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(s);
                });
            }
            bounds.s = dest[i].y;
            resetDestFuture();
            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(w);
                });
            }
            bounds.w = dest[i].x;
            resetDestFuture();
            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(e);
                });
            }
            bounds.e = dest[i].x;
            resetDestFuture();
            return bounds;
        };
        Square.prototype.insert = function (dim, index, data) {
            if (dim == 0 /* X */ && index > this.gridw || dim == 1 /* Y */ && index > this.gridh)
                return;
            if (data.length != (dim == 0 /* X */ ? this.gridh : this.gridw))
                return;
            console.log("before add: " + this.grid);
            switch (dim) {
                case 0 /* X */:
                    var newgrid = [];
                    for (var y = 0; y < this.gridh; y++) {
                        for (var x = 0; x < this.gridw + 1; x++) {
                            if (x > index) {
                                newgrid.push(this.get(new CartesianCoords(x - 1, y)));
                            }
                            else if (x == index) {
                                newgrid.push(data[y]);
                            }
                            else {
                                newgrid.push(this.get(new CartesianCoords(x, y)));
                            }
                        }
                    }
                    this.grid = newgrid;
                    this.gridw++;
                    break;
                case 1 /* Y */:
                    Array.prototype.splice.apply(this.grid, [this.gridw * index, 0].concat(data));
                    this.gridh++;
                    break;
            }
            this.size = this.gridw * this.gridh;
            console.log("after add: " + this.grid);
        };
        Square.prototype.remove = function (dim, index) {
            var _this = this;
            switch (dim) {
                case 0 /* X */:
                    console.log("removing column " + index);
                    this.grid = this.grid.filter(function (_, i) {
                        var c = _this.toCoords(i);
                        return c.x != index;
                    });
                    this.gridw--;
                    break;
                case 1 /* Y */:
                    console.log("removing row " + index);
                    this.grid = this.grid.filter(function (_, i) {
                        var c = _this.toCoords(i);
                        return c.y != index;
                    });
                    this.gridh--;
                    break;
            }
            this.size = this.gridw * this.gridh;
        };
        Square.prototype.genHalf = function (startY, endY, min, max) {
            var count = [];
            for (var i = startY * this.gridw; i < (endY) * this.gridw; i++) {
                var val = Math.round(Math.random() * (max - min) + min);
                this.setFlat(i, new Tile(2 /* REGULAR */, val));
                if (count[val] == null) {
                    count[val] = 0;
                }
                count[val]++;
            }
            for (i = startY * this.gridw; i < endY * this.gridw; i++) {
                if (Math.random() > 0.4) {
                    if (count[this.getFlat(i).value] > this.getFlat(i).value) {
                        count[this.getFlat(i).value]--;
                        this.setFlat(i, new Tile(1 /* EMPTY */, -1));
                    }
                }
            }
            for (i = min; i <= max; i++) {
                if (count[i] > i) {
                    for (var j = 0; j < this.getTileArray().length && count[i] > i; j++) {
                        if (this.getFlat(j).value == i) {
                            this.setFlat(j, new Tile(1 /* EMPTY */, -1));
                            count[i]--;
                        }
                    }
                }
            }
            for (i = 2; i <= max; i++) {
                if (count[i] < i) {
                    while (count[i] < i) {
                        var randIndex = Math.round(Math.random() * ((endY - startY) * this.gridw)) + startY * this.gridw;
                        if (this.isEmpty(randIndex)) {
                            this.setFlat(randIndex, new Tile(2 /* REGULAR */, i));
                            count[i]++;
                        }
                    }
                }
            }
        };
        Square.prototype.procGenGrid = function (min, max) {
            for (var y = 0; y < this.gridh; y++) {
                for (var x = 0; x < this.gridw; x++) {
                    var coords = new CartesianCoords(x, y);
                    var val = Math.round(Math.random() * (max - min) + min);
                    this.set(coords, new Tile(2 /* REGULAR */, val));
                }
            }
        };
        Square.prototype.debugPrint = function (plain) {
            var _this = this;
            if (plain === void 0) { plain = false; }
            if (plain) {
                var str = "";
                this.grid.forEach(function (t, i) {
                    if (i % _this.gridw == 0 && i != 0) {
                        str += "\n";
                    }
                    str += (t.isTangible() ? t.value.toString() : " ") + " ";
                });
                return str;
            }
            else {
                var str = "";
                {
                    str += "+";
                    for (var i = 0; i < this.gridw; i++)
                        str += "--";
                    str = str.substring(0, str.length - 1);
                    str += "+\n";
                }
                {
                    str += "|";
                    this.grid.forEach(function (t, i) {
                        if (i % _this.gridw == 0 && i != 0)
                            str += "|";
                        str += (t.isTangible() ? t.value.toString() : " ");
                        if (i % _this.gridw < _this.gridw - 1)
                            str += " ";
                        if (i % _this.gridw == _this.gridw - 1)
                            str += "|\n";
                    });
                }
                {
                    str += "+";
                    for (var i = 0; i < this.gridw; i++)
                        str += "--";
                    str = str.substring(0, str.length - 1);
                    str += "+";
                }
                return str;
            }
        };
        Square.prototype.debugPrintForLevelEditor = function () {
            var _this = this;
            var str = "[ ";
            this.grid.forEach(function (t, i) {
                if (i != 0 && i % _this.gridw == 0) {
                    str += "\n";
                }
                str += (t.isTangible() ? t.value.toString() : "0") + ", ";
            });
            if (this.grid.length > 0)
                str = str.substring(0, str.length - 2);
            str += " ]";
            return str;
        };
        return Square;
    })();
    Model.Square = Square;
})(Model || (Model = {}));
var SquareDim = (function () {
    function SquareDim() {
    }
    return SquareDim;
})();
var AnimationType;
(function (AnimationType) {
    AnimationType[AnimationType["SLIDE"] = 0] = "SLIDE";
    AnimationType[AnimationType["POP"] = 1] = "POP";
    AnimationType[AnimationType["MATERIALIZE"] = 2] = "MATERIALIZE";
    AnimationType[AnimationType["MELT"] = 3] = "MELT";
    AnimationType[AnimationType["CRUSH"] = 4] = "CRUSH";
    AnimationType[AnimationType["RESET"] = 5] = "RESET";
})(AnimationType || (AnimationType = {}));
var AnimationData = (function () {
    function AnimationData(type, params) {
        this.type = type;
        this.params = params;
    }
    AnimationData.prototype.toString = function () {
        return AnimationType[this.type];
    };
    return AnimationData;
})();
var Rect = (function () {
    function Rect(x, y, w, h) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }
    return Rect;
})();
var Colorizer = (function () {
    function Colorizer() {
        this.highlightFromTile = function (t) {
            if (t.type == 1 /* EMPTY */) {
                return this.scale[t.type](t.value / 9).hex();
            }
            else {
                return this.scale[t.type](t.value / 9).brighter().hex();
            }
        };
        this.foregroundFromColor = function (c) {
            return chroma(c).lab()[0] > 70 ? '#000' : '#fff';
        };
        this.scale = {};
        this.scale[1 /* EMPTY */] = chroma.scale(['#FFF', '#FFBDD8', '#B5D8EB', '#FFC8BA']);
        this.scale[2 /* REGULAR */] = chroma.scale(['#CCC', 'FFABAB', '#FBCA04', '#1DE5A2', '#3692B9', '#615258', '#EB6420', '#C8FF00', '#46727F', '#1D1163']);
        this.scale[4 /* LAVA */] = chroma.scale(['#AE5750', '#F96541', '#FF7939']);
    }
    Colorizer.prototype.fromTile = function (t) {
        if (t.type == 1 /* EMPTY */) {
            return this.scale[1 /* EMPTY */](0).hex();
        }
        else {
            return this.scale[t.type](t.value / 9).hex();
        }
    };
    return Colorizer;
})();
var Vec2 = (function () {
    function Vec2(x, y) {
        this.x = x;
        this.y = y;
    }
    Vec2.prototype.toString = function () {
        return this.x + "," + this.y;
    };
    return Vec2;
})();
var View;
(function (View) {
    function buildCanvas(backend) {
        switch (backend) {
            case 0 /* CANVAS */:
                break;
            case 1 /* SVG */:
                var vpDims = Utils.getViewport();
                return SVG('screen').size(vpDims[0], vpDims[1]).attr("preserveAspectRatio", "none");
                break;
        }
    }
    View.buildCanvas = buildCanvas;
    function resetCanvas(canvas, backend) {
        switch (backend) {
            case 0 /* CANVAS */:
                break;
            case 1 /* SVG */:
                break;
        }
    }
    View.resetCanvas = resetCanvas;
    function build(grid, backend, canvas, gameSignals) {
        return new SquareView(grid, backend, canvas, gameSignals);
    }
    View.build = build;
    function computePlayableRect() {
        var browserViewportDims = Utils.getViewport();
        var screenDim = new TSM.vec2(browserViewportDims);
        var small = Math.min.apply(Math, browserViewportDims);
        var playableDim = new TSM.vec2([small, small]);
        var centerOffset = screenDim.subtract(playableDim).scale(0.5);
        return new Rect(centerOffset.x, centerOffset.y, playableDim.x, playableDim.y);
    }
    View.computePlayableRect = computePlayableRect;
    var SquareView = (function () {
        function SquareView(square, backend, canvas, gameSignals) {
            this.drawBackend = backend;
            this.model = square;
            this.colorizer = new Colorizer();
            this.playableRect = computePlayableRect();
            this.cellSz = this.computeCellSize(this.playableRect);
            this.animQ = [];
            this.viewSignals = {
                animationFinished: new signals.Signal(),
                allFinished: new signals.Signal()
            };
            this.viewSignals.animationFinished.add(this.animationFinished, this);
            this.animating = false;
            this.concurrentAnimations = 0;
        }
        SquareView.prototype.resizeCanvas = function (canvas) {
            var vp = new TSM.vec2(Utils.getViewport());
            canvas.size(vp.x, vp.y);
            this.playableRect = computePlayableRect();
            this.cellSz = this.computeCellSize(this.playableRect);
        };
        SquareView.prototype.getCellSize = function () {
            return this.cellSz;
        };
        SquareView.prototype.computeCellSize = function (playableRect) {
            var margin = 20;
            var cellw = Math.floor((playableRect.w - margin) / this.model.maxgridw), cellh = Math.floor((playableRect.h - margin) / this.model.maxgridh);
            if (cellh != cellw) {
                cellh = cellw = Math.min(cellw, cellh);
            }
            if (cellw % 2 != 0)
                cellw--;
            if (cellh % 2 != 0)
                cellh--;
            var d = { w: cellw, h: cellh };
            return d;
        };
        SquareView.prototype.getGridCoordsFromScreenPos = function (canvas, pos) {
            var size = this.getCellSize();
            var cellw = size.w, cellh = size.h;
            pos = pos.subtract(new TSM.vec2([this.playableRect.x, this.playableRect.y])).subtract(this.gridOffset);
            return new CartesianCoords(Math.floor(pos.x / cellw), Math.floor(pos.y / cellh));
        };
        SquareView.prototype.drawEmptyBacking = function (canvas, x, y, e) {
            var cellw = this.getCellSize().w, cellh = this.getCellSize().h;
            var xOffset = cellw / 2 + Math.floor(this.playableRect.x + this.gridOffset.x);
            var yOffset = cellh / 2 + Math.floor(this.playableRect.y + this.gridOffset.y);
            var cell;
            if (this.drawBackend == 1 /* SVG */) {
                cell = canvas.group().transform({ x: x * cellw + xOffset, y: y * cellh + yOffset });
                var pts = [new Vec2(-cellw / 2, -cellh / 2), new Vec2(cellw / 2, -cellh / 2), new Vec2(cellw / 2, cellh / 2), new Vec2(-cellw / 2, cellh / 2)];
                var ptstr = pts.reduce(function (p1, p2, i, v) {
                    return p1.toString() + " " + p2.toString();
                }, "");
                var rect = cell.polygon(ptstr);
                cell.coords = new CartesianCoords(x, y);
                cell.rect = rect;
            }
            else if (this.drawBackend == 0 /* CANVAS */) {
            }
            return cell;
        };
        SquareView.prototype.copyStyle = function (src, dst) {
            var s = src;
            var d = dst;
            d.e = Utils.deepCopy(s.e);
            d.rect.attr({ 'fill': s.rect.attr('fill') });
        };
        SquareView.prototype.drawTile = function (canvas, x, y, e, dragOffset) {
            var cellw = this.getCellSize().w, cellh = this.getCellSize().h;
            var xOffset = cellw / 2 + Math.floor(this.playableRect.x + this.gridOffset.x);
            var yOffset = cellh / 2 + Math.floor(this.playableRect.y + this.gridOffset.y);
            var cell;
            if (this.drawBackend == 1 /* SVG */) {
                cell = canvas.group().transform({ x: x * cellw + xOffset + dragOffset.x, y: y * cellh + yOffset + dragOffset.y }).attr({ 'cursor': e.isTangible() ? 'move' : 'default' });
                var pts = [new Vec2(-cellw / 2, -cellh / 2), new Vec2(cellw / 2, -cellh / 2), new Vec2(cellw / 2, cellh / 2), new Vec2(-cellw / 2, cellh / 2)];
                var ptstr = pts.reduce(function (p1, p2, i, v) {
                    return p1.toString() + " " + p2.toString();
                }, "");
                var rect = cell.polygon(ptstr);
                rect.attr({ 'fill': this.colorizer.fromTile(e), 'fill-opacity': e.isTangible() ? 1 : 0 });
                cell.coords = new CartesianCoords(x, y);
                cell.rect = rect;
                cell.e = e;
                cell.cannonicalTransform = { x: x * cellw + xOffset, y: y * cellh + yOffset };
                cell.translateTarget = { x: x * cellw + xOffset, y: y * cellh + yOffset };
                cell.dragOffset = dragOffset;
            }
            else if (this.drawBackend == 0 /* CANVAS */) {
            }
            return cell;
        };
        SquareView.prototype.resetView = function (canvas) {
            console.log("resetView");
            if (this.model == null)
                return;
            if (this.cells == null)
                this.cells = [];
            if (this.emptyBackings == null)
                this.emptyBackings = [];
            switch (this.drawBackend) {
                case 0 /* CANVAS */:
                    break;
                case 1 /* SVG */:
                    if (this.cells.length != this.model.size) {
                        canvas.clear();
                    }
                    break;
            }
            var cellSz = this.getCellSize();
            var gridDim = new TSM.vec2([cellSz.w * this.model.gridw, cellSz.h * this.model.gridh]);
            this.gridOffset = new TSM.vec2([this.playableRect.w, this.playableRect.h]).subtract(gridDim).scale(0.5);
            {
                var stroke = 2;
                var tl = (Math.floor(this.playableRect.x + this.gridOffset.x) - stroke / 2) + " " + (Math.floor(this.playableRect.y + this.gridOffset.y) - stroke / 2);
                var tr = (Math.floor(this.playableRect.x + this.gridOffset.x) + gridDim.x + stroke / 2) + " " + (Math.floor(this.playableRect.y + this.gridOffset.y) - stroke / 2);
                var br = (Math.floor(this.playableRect.x + this.gridOffset.x) + gridDim.x + stroke / 2) + " " + (Math.floor(this.playableRect.y + this.gridOffset.y) + gridDim.y + stroke / 2);
                var bl = (Math.floor(this.playableRect.x + this.gridOffset.x) - stroke / 2) + " " + (Math.floor(this.playableRect.y + this.gridOffset.y) + gridDim.y + stroke / 2);
                if (this.border != null)
                    this.border.remove();
                this.border = canvas.path("M  " + tl + " L " + tr + " L " + br + " L " + bl + " z").attr({ "fill": "none", "stroke": "#000", "stroke-width": stroke });
            }
            if (this.drawBackend == 1 /* SVG */) {
                if (this.bg != null) {
                    this.bg.remove();
                }
                this.bg = canvas.rect(gridDim.x, gridDim.y);
                this.bg.attr({ "x": Math.floor(this.playableRect.x + this.gridOffset.x) + stroke / 2, "y": Math.floor(this.playableRect.y + this.gridOffset.y) + stroke / 2, "width": gridDim.x - stroke / 2, "height": gridDim.y - stroke / 2, "fill": '#fff', "fill-opacity": 1 });
            }
            if (this.clipper == null) {
                this.clipper = canvas.clip().rect(gridDim.x, gridDim.y).move(this.playableRect.x + this.gridOffset.x, this.playableRect.y + this.gridOffset.y);
            }
            this.board = canvas.group();
            for (var y = 0; y < this.model.gridh; y++) {
                for (var x = 0; x < this.model.gridw; x++) {
                    var e = this.model.get(new CartesianCoords(x, y));
                    if (this.emptyBackings[this.model.toFlat(x, y)] != null)
                        this.emptyBackings[this.model.toFlat(x, y)].remove();
                    this.emptyBackings[this.model.toFlat(x, y)] = this.drawEmptyBacking(this.board, x, y, e);
                }
            }
            for (var y = 0; y < this.model.gridh; y++) {
                for (var x = 0; x < this.model.gridw; x++) {
                    var e = this.model.get(new CartesianCoords(x, y));
                    var offs = new TSM.vec2([0, 0]);
                    if (this.cells[this.model.toFlat(x, y)] != null) {
                        this.cells[this.model.toFlat(x, y)].remove();
                    }
                    var cell = this.drawTile(this.board, x, y, e, offs);
                    this.cells[this.model.toFlat(x, y)] = cell;
                }
            }
            this.board.clipWith(this.clipper);
        };
        SquareView.prototype.getViewElements = function () {
            return this.cells;
        };
        SquareView.prototype.getViewElement = function (c) {
            return Math.abs(Math.round(c.y) - c.y) <= Utils.EPSILON && Math.abs(Math.round(c.x) - c.x) <= Utils.EPSILON && c.y >= 0 && c.x >= 0 && c.x < this.model.gridw && c.y < this.model.gridh ? this.cells[this.model.toFlat(c.x, c.y)] : null;
        };
        SquareView.prototype.setViewElement = function (c, e) {
            if (Math.abs(Math.round(c.y) - c.y) <= Utils.EPSILON && Math.abs(Math.round(c.x) - c.x) <= Utils.EPSILON && c.y >= 0 && c.x >= 0 && c.x < this.model.gridw && c.y < this.model.gridh) {
                this.cells[this.model.toFlat(c.x, c.y)] = e;
            }
        };
        SquareView.prototype.getEmptyBackingViewElement = function (c) {
            return Math.abs(Math.round(c.y) - c.y) <= Utils.EPSILON && Math.abs(Math.round(c.x) - c.x) <= Utils.EPSILON && c.y >= 0 && c.x >= 0 && c.x < this.model.gridw && c.y < this.model.gridh ? this.emptyBackings[this.model.toFlat(c.x, c.y)] : null;
        };
        SquareView.prototype.getClosestLowerViewElement = function (c) {
            var cl = new CartesianCoords(Math.floor(c.x), Math.floor(c.y));
            return this.getViewElement(cl);
        };
        SquareView.prototype.getClosestUpperViewElement = function (c) {
            var cu = new CartesianCoords(Math.ceil(c.x), Math.ceil(c.y));
            return this.getViewElement(cu);
        };
        SquareView.prototype.debugPrint = function () {
            var _this = this;
            var str = "";
            this.cells.forEach(function (c, i) {
                if (i % _this.model.gridw == 0) {
                    str += "\n";
                }
                str += (c != null && c.e.isTangible() ? c.e.value.toString() : " ") + " ";
            });
            return str;
        };
        SquareView.prototype.physicalSlide = function (canvas, group, blockDistance, doneCallback) {
            var _this = this;
            if (this.drawBackend == 1 /* SVG */) {
                var speed = 1.5;
                var d = Math.sqrt(Math.pow(blockDistance.x * this.cellSz.w, 2) + Math.pow(blockDistance.y * this.cellSz.h, 2));
                var anchor = canvas.group();
                anchor.animate(d / speed, '>', 0).during(function (t) {
                    group.forEach(function (f, i) {
                        var fromElement = _this.getViewElement(f);
                        var emptyBacking = _this.getEmptyBackingViewElement(f);
                        if (fromElement == null)
                            return;
                        var x = fromElement.cannonicalTransform.x + (blockDistance.x * _this.cellSz.w) * SVG.easing.quadOut(t);
                        var y = fromElement.cannonicalTransform.y + (blockDistance.y * _this.cellSz.h) * SVG.easing.quadOut(t);
                        fromElement.translate(x, y);
                    });
                }).after(function () {
                    anchor.remove();
                    doneCallback();
                    _this.viewSignals.animationFinished.dispatch();
                });
            }
        };
        SquareView.prototype.slide = function (canvas, from, to, doneCallback) {
            var _this = this;
            if (this.drawBackend == 1 /* SVG */) {
                var e1 = this.getViewElement(from[0]);
                var e2 = this.getViewElement(to[0]);
                var isTweenElement = false;
                var e2a = null;
                var e2b = null;
                if (e2 == null) {
                    isTweenElement = true;
                    e2a = this.getClosestLowerViewElement(to[0]);
                    e2b = this.getClosestUpperViewElement(to[0]);
                    if (e2a == null || e2b == null)
                        debugger;
                }
                if (e1 == null)
                    debugger;
                var d = 0;
                if (isTweenElement) {
                    d = Math.abs(e2a.transform('x') / 2 + e2b.transform('x') / 2 - e1.transform('x')) + Math.abs(e2a.transform('y') / 2 + e2b.transform('y') / 2 - e1.transform('y'));
                }
                else {
                    d = Math.abs(e2.transform('x') - e1.transform('x')) + Math.abs(e2.transform('y') - e1.transform('y'));
                }
                var speed = 1.5;
                from.forEach(function (f, i) {
                    var fromElement = _this.getViewElement(f);
                    var toElement = _this.getViewElement(to[i]);
                    var toElement1 = null;
                    var toElement2 = null;
                    if (isTweenElement) {
                        toElement1 = _this.getClosestLowerViewElement(to[i]);
                        toElement2 = _this.getClosestUpperViewElement(to[i]);
                        fromElement.translateTarget.x += toElement1.cannonicalTransform.x / 2 + toElement2.cannonicalTransform.x / 2 - fromElement.cannonicalTransform.x;
                        fromElement.translateTarget.y += toElement1.cannonicalTransform.y / 2 + toElement2.cannonicalTransform.y / 2 - fromElement.cannonicalTransform.y;
                    }
                    else {
                        fromElement.translateTarget.x = toElement.cannonicalTransform.x;
                        fromElement.translateTarget.y = toElement.cannonicalTransform.y;
                    }
                });
                var anchor = canvas.group();
                anchor.animate(d / speed, '>', 0).during(function (t) {
                    from.forEach(function (f, i) {
                        var fromElement = _this.getViewElement(f);
                        var emptyBacking = _this.getEmptyBackingViewElement(f);
                        if (fromElement == null)
                            return;
                        var x = fromElement.cannonicalTransform.x + fromElement.dragOffset.x + (fromElement.translateTarget.x - fromElement.cannonicalTransform.x - fromElement.dragOffset.x) * SVG.easing.quadOut(t);
                        var y = fromElement.cannonicalTransform.y + fromElement.dragOffset.y + (fromElement.translateTarget.y - fromElement.cannonicalTransform.y - fromElement.dragOffset.y) * SVG.easing.quadOut(t);
                        fromElement.translate(x, y);
                        if (isTweenElement)
                            emptyBacking.translate(x, y);
                    });
                }).after(function () {
                    var fromElements = from.map(function (f) {
                        return _this.getViewElement(f);
                    });
                    fromElements.forEach(function (fromElement, i) {
                        if (fromElement == null)
                            return;
                        fromElement.dragOffset.x = 0;
                        fromElement.dragOffset.y = 0;
                        fromElement.translateTarget.x = fromElement.transform('x');
                        fromElement.translateTarget.y = fromElement.transform('y');
                        fromElement.cannonicalTransform = _this.getViewElement(to[i]).cannonicalTransform;
                    });
                    to.forEach(function (t, i) {
                        var toElement = _this.getViewElement(t);
                        if (!to.some(function (h) {
                            return h.equals(from[i]);
                        })) {
                            _this.setViewElement(from[i], null);
                        }
                        _this.setViewElement(t, fromElements[i]);
                    });
                    anchor.remove();
                    _this.viewSignals.animationFinished.dispatch();
                    if (doneCallback != null)
                        doneCallback();
                });
            }
            else if (this.drawBackend == 0 /* CANVAS */) {
            }
        };
        SquareView.prototype.crush = function (canvas, c, dir, doneCallback) {
            var _this = this;
            var cell = this.getViewElement(c);
            var cellh = this.getCellSize().h;
            var horiz = dir == 3 /* WEST */ || dir == 1 /* EAST */;
            var vert = !horiz;
            if (this.drawBackend == 1 /* SVG */) {
                cell.animate(100, '<', 0).scale(horiz ? 0 : 1, vert ? 0 : 1).move(horiz ? (cell.cannonicalTransform.x + (dir == 3 /* WEST */ ? -1 : 1) * cellh / 2) : cell.cannonicalTransform.x, vert ? (cell.cannonicalTransform.y + (dir == 0 /* NORTH */ ? -1 : 1) * cellh / 2) : cell.cannonicalTransform.y).after(function () {
                    _this.viewSignals.animationFinished.dispatch();
                    if (doneCallback != null)
                        doneCallback();
                });
            }
        };
        SquareView.prototype.melt = function (canvas, dim, index, doneCallback) {
            var _this = this;
            switch (dim) {
                case 0 /* X */:
                    var grp = this.emptyBackings.filter(function (_, i) {
                        var c = _this.model.toCoords(i);
                        return c.x == index;
                    });
                    var target = canvas.group();
                    target.animate(250, '>', 0).during(function (t) {
                        var s = 1 - SVG.easing.quadOut(t);
                        grp.forEach(function (c, i) {
                            c.scale(s, s);
                        });
                    }).after(function () {
                        _this.viewSignals.animationFinished.dispatch();
                        target.remove();
                        if (doneCallback != null)
                            doneCallback();
                    });
                    break;
                case 1 /* Y */:
                    var grp = this.emptyBackings.filter(function (_, i) {
                        var c = _this.model.toCoords(i);
                        return c.y == index;
                    });
                    var target = canvas.group();
                    target.animate(250, '>', 0).during(function (t) {
                        var s = 1 - SVG.easing.quadOut(t);
                        grp.forEach(function (c, i) {
                            c.scale(s, s);
                        });
                    }).after(function () {
                        _this.viewSignals.animationFinished.dispatch();
                        target.remove();
                        if (doneCallback != null)
                            doneCallback();
                    });
                    break;
            }
        };
        SquareView.prototype.pop = function (canvas, cs, doneCallback) {
            var _this = this;
            var target = canvas.group();
            target.animate(200, '>', 0).during(function (t) {
                cs.forEach(function (c, i) {
                    var element = _this.getViewElement(c);
                    if (_this.drawBackend == 1 /* SVG */) {
                        var scaleX = 1 - SVG.easing.quadOut(t);
                        var scaleY = 1 - SVG.easing.quadOut(t);
                        element.scale(scaleX, scaleY);
                    }
                    else if (_this.drawBackend == 0 /* CANVAS */) {
                    }
                });
            }).after(function () {
                _this.viewSignals.animationFinished.dispatch();
                if (doneCallback != null)
                    doneCallback();
                target.remove();
            });
        };
        SquareView.prototype.materialize = function (canvas, cs, doneCallback) {
            var _this = this;
            canvas.animate(200, '>', 0).during(function (t) {
                cs.forEach(function (c, i) {
                    var element = _this.getViewElement(c);
                    if (_this.drawBackend == 1 /* SVG */) {
                        var scaleX = SVG.easing.quadOut(t);
                        var scaleY = SVG.easing.quadOut(t);
                        element.scale(scaleX, scaleY);
                    }
                    else if (_this.drawBackend == 0 /* CANVAS */) {
                    }
                });
            }).after(function () {
                _this.viewSignals.animationFinished.dispatch();
                doneCallback();
            });
        };
        SquareView.prototype.indicateTurn = function (canvas, yourTurn) {
            var target = canvas.group();
            target.attr({ 'opacity': 0 });
            var text = null;
            var cellSz = this.getCellSize();
            var gridDim = new TSM.vec2([cellSz.w * this.model.gridw, cellSz.h * this.model.gridh]);
            if (yourTurn) {
                text = target.plain("Your Turn");
                target.translate(this.playableRect.x + this.gridOffset.x + gridDim.x / 2, this.playableRect.y + this.gridOffset.y + gridDim.y + 40);
            }
            else {
                text = target.plain("Opponent's Turn");
                target.translate(this.playableRect.x + this.gridOffset.x + gridDim.x / 2, this.playableRect.y + this.gridOffset.y - 30);
            }
            text.attr({ 'fill': '#000', 'font-size': 30, 'text-anchor': 'middle' });
            target.animate(1000, '>', 0).during(function (t) {
                var opacity = SVG.easing.quadIn(t);
                target.attr({ 'opacity': opacity });
            }).after(function () {
                target.animate(1000, '>', 1500).during(function (t) {
                    var opacity = 1 - SVG.easing.quadOut(t);
                    target.attr({ 'opacity': opacity });
                }).after(function () {
                    target.remove();
                });
            });
        };
        SquareView.prototype.queueAnimation = function (anims) {
            console.log("received signal for animations: " + anims);
            if (this.animQ.length == 0 && !this.animating) {
                console.log("playing...");
                this.playAnimations(anims);
            }
            else {
                console.log("queueing...");
                this.animQ.push(anims);
            }
        };
        SquareView.prototype.playAnimations = function (anims) {
            var _this = this;
            this.animating = true;
            anims.forEach(function (a) {
                _this.concurrentAnimations++;
                switch (a.type) {
                    case 0 /* SLIDE */:
                        _this.slide.apply(_this, a.params);
                        break;
                    case 1 /* POP */:
                        _this.pop.apply(_this, a.params);
                        break;
                    case 2 /* MATERIALIZE */:
                        _this.materialize.apply(_this, a.params);
                        break;
                    case 3 /* MELT */:
                        _this.melt.apply(_this, a.params);
                        break;
                    case 4 /* CRUSH */:
                        _this.crush.apply(_this, a.params);
                        break;
                }
            });
        };
        SquareView.prototype.animationFinished = function () {
            this.concurrentAnimations--;
            if (this.concurrentAnimations == 0)
                this.animating = false;
            if (this.animQ.length > 0) {
                var next = this.animQ.shift();
                if (next != undefined) {
                    this.playAnimations(next);
                }
            }
            else {
                this.viewSignals.allFinished.dispatch();
            }
        };
        return SquareView;
    })();
    View.SquareView = SquareView;
})(View || (View = {}));
var ui;
var UIStack = (function () {
    function UIStack() {
        this.stack = [];
    }
    UIStack.prototype.push = function (layer) {
        if (this.stack[this.stack.length - 1] != null)
            this.stack[this.stack.length - 1].leave();
        this.stack.push(layer);
        layer.enter();
    };
    UIStack.prototype.pop = function () {
        var layer = this.stack.pop();
        layer.leave();
    };
    return UIStack;
})();
ui = new UIStack();
var Border;
(function (Border) {
    Border[Border["TOP"] = 0] = "TOP";
    Border[Border["BOTTOM"] = 1] = "BOTTOM";
    Border[Border["RIGHT"] = 2] = "RIGHT";
    Border[Border["LEFT"] = 3] = "LEFT";
})(Border || (Border = {}));
;
var SquareGameState = (function () {
    function SquareGameState() {
    }
    return SquareGameState;
})();
var SquareGame = (function () {
    function SquareGame(gameType, drawBackend, level) {
        var _this = this;
        this.gameSignals = {
            updated: new signals.Signal()
        };
        this.timeSinceLastUpdate = new Date().getTime();
        this.canvas = View.buildCanvas(drawBackend);
        if (drawBackend == 1 /* SVG */) {
            var h = new Hammer(this.canvas.node, { preventDefault: true });
            h.get('pan').set({ direction: Hammer.DIRECTION_ALL });
            h.get('swipe').set({ direction: Hammer.DIRECTION_ALL });
            h.on("swipe panend", function (e) {
                _this.resolveDrag(e);
            });
            h.on("panstart", function (e) {
                _this.startDrag(e);
            });
            h.on("panmove", function (e) {
                _this.midDrag(e);
            });
            h.on("press", function (e) {
                console.log("press");
            });
        }
    }
    SquareGame.prototype.updateViewAndController = function () {
        this.view.resetView(this.canvas);
        this.extendUIFromView();
    };
    SquareGame.prototype.extendUIOnce = function () {
        var _this = this;
        if (this.view.drawBackend == 0 /* CANVAS */) {
        }
        else {
            Events.unbind(window, "resize");
            Events.bind(window, "resize", function () {
                console.log("resizing...");
                _this.view.resizeCanvas(_this.canvas);
                _this.updateViewAndController();
            });
        }
    };
    SquareGame.prototype.extendUIFromView = function () {
        var _this = this;
        if (this.gameParams.drawBackend == 1 /* SVG */) {
            var cells = this.view.getViewElements();
            cells.forEach(function (cell, i) {
                if (cell === null)
                    return;
                cell.mouseover(null).mouseout(null);
                cell.mouseover(function () {
                    if (_this.view.animating)
                        return;
                    if (_this.model.isEmpty(cell.coords))
                        return;
                    var hover = _this.model.floodAcquire(cell.coords);
                    hover.forEach(function (t) {
                        _this.view.getViewElement(t).rect.attr({ 'fill': _this.view.colorizer.highlightFromTile(_this.model.getFlat(i)) });
                    });
                }).mouseout(function () {
                    if (_this.view.animating)
                        return;
                    var hover = _this.model.floodAcquire(cell.coords);
                    hover.forEach(function (t) {
                        _this.view.getViewElement(t).rect.attr({ 'fill': _this.view.colorizer.fromTile(_this.model.getFlat(i)) });
                    });
                });
            });
        }
        else if (this.gameParams.drawBackend == 0 /* CANVAS */) {
        }
    };
    SquareGame.prototype.restart = function () {
        console.log("restarting...");
    };
    SquareGame.prototype.importGridState = function (state, gridw, gridh, maxgridw, maxgridh, flip) {
        if (flip === void 0) { flip = false; }
        console.log("new grid dims: " + gridw + " x " + gridh);
        this.model.import(state, flip);
        {
            var tr = this.tracker;
            tr.tiles = [];
            state.map(function (n) {
                if (tr.tiles[n] == undefined) {
                    tr.tiles[n] = 1;
                }
                else {
                    tr.tiles[n]++;
                }
            });
        }
        this.model.gridw = gridw;
        this.model.gridh = gridh;
        this.model.maxgridw = maxgridw;
        this.model.maxgridh = maxgridh;
        this.model.size = gridw * gridh;
        this.update();
    };
    SquareGame.prototype.init = function (gp) {
        var _this = this;
        console.log("init called");
        var gs = {
            dragStart: new TSM.vec2([0, 0]),
            dragDelta: new TSM.vec2([0, 0]),
            selected: [],
            lastCleared: 0,
            numMoves: 0,
            dragUpdateID: null,
            dragAnchor: null,
            lockedDragDirection: null
        };
        this.gameParams = gp;
        this.gameState = gs;
        var tr = {
            tiles: (function () {
                var t = [];
                for (var i = MIN_VAL; i <= gp.level.maxVal; i++) {
                    t[i] = 0;
                }
                return t;
            })()
        };
        this.tracker = tr;
        this.model = new Model.Square(this.gameParams.level.gridw, this.gameParams.level.gridh, this.gameParams.level.maxGridw, this.gameParams.level.maxGridh, new Tile(1 /* EMPTY */, 0), new Tile(0 /* OUT_OF_BOUNDS */, -1));
        if (gp.level.layout == null) {
            this.procGenGrid(this.model, gp, this.tracker);
        }
        else {
            this.model.import(gp.level.layout, false);
            {
                gp.level.layout.map(function (n) {
                    if (_this.tracker.tiles[n] == undefined) {
                        _this.tracker.tiles[n] = 1;
                    }
                    else {
                        _this.tracker.tiles[n]++;
                    }
                });
            }
        }
        if (gp.drawBackend == 1 /* SVG */)
            this.canvas.clear();
        View.resetCanvas(this.canvas, gp.drawBackend);
        this.view = View.build(this.model, gp.drawBackend, this.canvas, this.gameSignals);
        switch (this.gameParams.gameType) {
            case 0 /* SURVIVAL */:
                break;
            case 1 /* PUZZLE */:
                break;
        }
        if (this.updateID != null) {
            clearInterval(this.updateID);
        }
        this.connectModelSignals(this.model);
        this.connectViewSignals(this.view);
        this.updateID = setInterval(function () { return _this.timedUpdate(); }, 1000 / 5);
        this.updateViewAndController();
        this.extendUIOnce();
    };
    SquareGame.prototype.timedUpdate = function () {
        var now = new Date().getTime();
        var dt = now - this.timeSinceLastUpdate;
        this.timeSinceLastUpdate = now;
    };
    SquareGame.prototype.update = function () {
        console.log(this.model);
        if (this.gameSignals != null)
            this.gameSignals.updated.dispatch(this.debugPrintModelForLevelEditor(), this.model.gridw, this.model.gridh);
        this.model.recomputeAllBounds();
        if (this.clearedStage()) {
            if (this.gameParams.gameType == 0 /* SURVIVAL */) {
                alert("Holy crap you beat survival mode! How is that even possible. Let's see you do it again.");
                this.init(this.gameParams);
            }
            else {
            }
        }
        else if (this.over()) {
            console.log(this.model.size);
            alert("Better luck next time");
            this.init(this.gameParams);
        }
        else {
            this.updateViewAndController();
        }
    };
    SquareGame.prototype.procGenGrid = function (grid, gp, tr) {
        if (gp.gameType == 0 /* SURVIVAL */) {
            var done = false;
            var acc = 0;
            var toGenerate = [];
            while (!done) {
                var val = Math.round(Math.random() * (gp.level.maxVal - MIN_VAL) + MIN_VAL);
                while (toGenerate.some(function (v) {
                    return v == val;
                })) {
                    val = Math.round(Math.random() * (gp.level.maxVal - MIN_VAL) + MIN_VAL);
                }
                if (acc + val < grid.size - 2 * gp.level.gridw) {
                    toGenerate.push(val);
                    acc += val;
                }
                else {
                    done = true;
                }
            }
            while (toGenerate.length > 0) {
                var val = toGenerate.pop();
                var added = 0;
                while (added < val) {
                    var randIndex = Math.round(Math.random() * (grid.size - 1));
                    if (grid.isEmpty(randIndex)) {
                        grid.setFlat(randIndex, new Tile(2 /* REGULAR */, val));
                        added++;
                    }
                }
                tr.tiles[val] = val;
            }
            for (var y = 0; y < grid.gridh; y++) {
                for (var x = 0; x < grid.gridw; x++) {
                    var coords = new CartesianCoords(x, y);
                    var t = grid.get(coords);
                    if (this.model.floodAcquire(coords).length == t.value) {
                        do {
                            grid.set(coords, new Tile(1 /* EMPTY */, -1));
                            var newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            while (grid.get(newCoords).isTangible()) {
                                newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            }
                            grid.set(newCoords, t);
                            var canPop = this.model.floodAcquire(newCoords).length == t.value;
                        } while (canPop);
                    }
                }
            }
        }
        else {
            this.model.procGenGrid(MIN_VAL, gp.level.maxVal);
        }
    };
    SquareGame.prototype.justMove = function (group, direction, steps) {
        this.model.move(group, direction, steps);
        this.updateViewAndController();
    };
    SquareGame.prototype.dragUpdate = function () {
        var _this = this;
        var cellSize = this.view.getCellSize();
        var dragDir;
        if (Math.abs(this.gameState.dragDelta.y) > Math.abs(this.gameState.dragDelta.x)) {
            dragDir = this.gameState.dragDelta.y < 0 ? 0 /* NORTH */ : 2 /* SOUTH */;
        }
        else {
            dragDir = this.gameState.dragDelta.x < 0 ? 3 /* WEST */ : 1 /* EAST */;
        }
        var wrapCount = 0;
        switch (dragDir) {
            case 0 /* NORTH */:
                wrapCount = -Utils.truncate(this.gameState.dragDelta.y / cellSize.h);
                break;
            case 1 /* EAST */:
                wrapCount = Utils.truncate(this.gameState.dragDelta.x / cellSize.w);
                break;
            case 2 /* SOUTH */:
                wrapCount = Utils.truncate(this.gameState.dragDelta.y / cellSize.h);
                break;
            case 3 /* WEST */:
                wrapCount = -Utils.truncate(this.gameState.dragDelta.x / cellSize.w);
                break;
        }
        this.gameState.selected.forEach(function (coord, index) {
            var c = _this.view.getViewElement(coord);
            var x = c.cannonicalTransform.x;
            var y = c.cannonicalTransform.y;
            var dx = _this.gameState.dragDelta.x;
            var dy = _this.gameState.dragDelta.y;
            var wrapX = 0;
            var wrapY = 0;
            switch (dragDir) {
                case 0 /* NORTH */:
                    wrapY = cellSize.h * _this.model.gridh * (Math.ceil((wrapCount - index) / _this.gameState.selected.length));
                    break;
                case 1 /* EAST */:
                    wrapX = -cellSize.w * _this.model.gridw * (Math.ceil((wrapCount - _this.gameState.selected.length + 1 + index) / _this.gameState.selected.length));
                    break;
                case 2 /* SOUTH */:
                    wrapY = -cellSize.h * _this.model.gridh * (Math.ceil((wrapCount - _this.gameState.selected.length + 1 + index) / _this.gameState.selected.length));
                    break;
                case 3 /* WEST */:
                    wrapX = cellSize.w * _this.model.gridw * (Math.ceil((wrapCount - index) / _this.gameState.selected.length));
                    break;
            }
            c.translate(x + dx + wrapX, y + dy + wrapY);
            c.dragOffset.x = dx;
            c.dragOffset.y = dy;
            if (Math.abs(_this.gameState.dragDelta.x) > 0) {
                c.scale(1.005, 1);
            }
            else if (Math.abs(_this.gameState.dragDelta.y) > 0) {
                c.scale(1, 1.005);
            }
        });
        if (this.gameState.selected.length > 0) {
            var refIndex = 0;
            switch (dragDir) {
                case 0 /* NORTH */:
                case 3 /* WEST */:
                    refIndex = wrapCount % this.gameState.selected.length;
                    break;
                case 1 /* EAST */:
                case 2 /* SOUTH */:
                    refIndex = this.gameState.selected.length - (wrapCount % this.gameState.selected.length) - 1;
                    break;
            }
            var ref = this.view.getViewElement(this.gameState.selected[refIndex]);
            this.view.copyStyle(ref, this.phantomCell);
            var x = ref.transform('x');
            var y = ref.transform('y');
            switch (dragDir) {
                case 0 /* NORTH */:
                    this.phantomCell.translate(x, y + cellSize.h * this.model.gridh);
                    break;
                case 1 /* EAST */:
                    this.phantomCell.translate(x - cellSize.w * this.model.gridw, y);
                    break;
                case 2 /* SOUTH */:
                    this.phantomCell.translate(x, y - cellSize.h * this.model.gridh);
                    break;
                case 3 /* WEST */:
                    this.phantomCell.translate(x + cellSize.w * this.model.gridw, y);
                    break;
            }
        }
    };
    SquareGame.prototype.midDrag = function (e) {
        var _this = this;
        var moveVector = new CartesianCoords(0, 0);
        var moved = false;
        var deadzone = this.view.getCellSize().w / 8;
        if (Math.abs(e.deltaY) > deadzone || Math.abs(e.deltaX) > deadzone) {
            var moveDirection;
            if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
                moveDirection = e.deltaY < 0 ? 0 /* NORTH */ : 2 /* SOUTH */;
            }
            else {
                moveDirection = e.deltaX < 0 ? 3 /* WEST */ : 1 /* EAST */;
            }
            if (this.gameState.dragUpdateID == null) {
                this.gameState.dragUpdateID = setInterval(function () { return _this.dragUpdate(); }, 1000 / 60);
            }
            if (this.gameState.lockedDragDirection == null) {
                this.gameState.lockedDragDirection = moveDirection;
            }
            switch (this.gameState.lockedDragDirection) {
                case 0 /* NORTH */:
                case 2 /* SOUTH */:
                    this.gameState.dragDelta.x = 0;
                    this.gameState.dragDelta.y = e.deltaY;
                    this.gameState.selected = this.model.getCol(this.gameState.dragAnchor);
                    break;
                case 3 /* WEST */:
                case 1 /* EAST */:
                    this.gameState.dragDelta.x = e.deltaX;
                    this.gameState.dragDelta.y = 0;
                    this.gameState.selected = this.model.getRow(this.gameState.dragAnchor);
                    break;
            }
        }
        else {
            this.gameState.selected.forEach(function (coord) {
                var c = _this.view.getViewElement(coord);
                c.translate(c.cannonicalTransform.x, c.cannonicalTransform.y);
                c.dragOffset.x = 0;
                c.dragOffset.y = 0;
            });
            this.gameState.dragDelta.x = 0;
            this.gameState.dragDelta.y = 0;
        }
    };
    SquareGame.prototype.startDrag = function (e) {
        var ctr = new TSM.vec2([e.center.x, e.center.y]);
        var so = new TSM.vec2(Utils.getScrollOffset());
        so = so.negate();
        var offs = new TSM.vec2(Utils.getOffset(this.canvas.node));
        var screenPos = ctr.subtract(so).subtract(offs);
        var mouseCoords = this.view.getGridCoordsFromScreenPos(this.canvas, screenPos);
        if (this.model.get(mouseCoords).isTangible()) {
            this.gameState.selected = this.model.floodAcquire(mouseCoords);
            this.gameState.dragAnchor = mouseCoords;
        }
        else {
            this.gameState.selected = [];
            this.gameState.dragAnchor = null;
        }
        var cellSz = this.view.getCellSize();
        this.phantomCell = this.view.drawTile(this.view.board, this.model.gridw * cellSz.w, this.model.gridh * cellSz.h, new Tile(2 /* REGULAR */, 0), new TSM.vec2([0, 0]));
    };
    SquareGame.prototype.resolveDrag = function (e) {
        var _this = this;
        if (this.gameState.dragUpdateID != null) {
            clearInterval(this.gameState.dragUpdateID);
            this.gameState.dragUpdateID = null;
        }
        if (this.gameState.selected == null || this.gameState.selected.length == 0) {
            console.log("nothing selected; or trying to move tiles that don't belong to you...");
            return;
        }
        var moveDirection;
        if (this.gameState.dragDelta.squaredLength() > 0) {
            if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
                moveDirection = e.deltaY < 0 ? 0 /* NORTH */ : 2 /* SOUTH */;
            }
            else {
                moveDirection = e.deltaX < 0 ? 3 /* WEST */ : 1 /* EAST */;
            }
            var wrapCount = 0;
            var cellSize = this.view.getCellSize();
            switch (moveDirection) {
                case 0 /* NORTH */:
                    wrapCount = -Math.floor(this.gameState.dragDelta.y / cellSize.h);
                    break;
                case 1 /* EAST */:
                    wrapCount = Math.ceil(this.gameState.dragDelta.x / cellSize.w);
                    break;
                case 2 /* SOUTH */:
                    wrapCount = Math.ceil(this.gameState.dragDelta.y / cellSize.h);
                    break;
                case 3 /* WEST */:
                    wrapCount = -Math.floor(this.gameState.dragDelta.x / cellSize.w);
                    break;
            }
            this.justMove(this.gameState.selected, moveDirection, wrapCount);
        }
        this.gameState.selected.forEach(function (coord) {
            var c = _this.view.getViewElement(coord);
            c.scale(1, 1);
        });
        if (this.phantomCell != null) {
            this.phantomCell.remove();
            this.phantomCell = null;
        }
        this.gameState.lockedDragDirection = null;
    };
    SquareGame.prototype.over = function () {
        return false;
    };
    SquareGame.prototype.clearedStage = function () {
        return false;
    };
    SquareGame.prototype.advance = function () {
    };
    SquareGame.prototype.crease = function () {
        var _this = this;
        var colsToRemove = [];
        for (var col = 0; col < this.model.gridw; col++) {
            var isColumnEmpty = true;
            for (var y = 0; y < this.model.gridh; y++) {
                if (!this.model.isEmpty(new CartesianCoords(col, y))) {
                    isColumnEmpty = false;
                }
            }
            if (isColumnEmpty)
                colsToRemove.push(col);
        }
        var rowsToRemove = [];
        for (var row = 0; row < this.model.gridh; row++) {
            var isRowEmpty = true;
            for (var x = 0; x < this.model.gridw; x++) {
                if (!this.model.isEmpty(new CartesianCoords(x, row))) {
                    isRowEmpty = false;
                }
            }
            if (isRowEmpty)
                rowsToRemove.push(row);
        }
        if (rowsToRemove.length > 0 || colsToRemove.length > 0) {
            var rowsToPrune = [];
            rowsToRemove.forEach(function (v, i, rs) {
                console.log("v:" + v + ", i:" + i);
                if (rowsToPrune.indexOf(v - i) == -1 && v < _this.model.gridh - 1) {
                    rowsToPrune.push(v - i);
                }
            });
            var colsToPrune = [];
            colsToRemove.forEach(function (v, i, cs) {
                console.log("v:" + v + ", i:" + i);
                if (colsToPrune.indexOf(v - i) == -1 && v < _this.model.gridw - 1) {
                    colsToPrune.push(v - i);
                }
            });
            rowsToRemove.reverse();
            colsToRemove.reverse();
            var doPostPrune = function () {
                console.log("rows to prune: ");
                console.log(rowsToPrune);
                console.log("cols to prune: ");
                console.log(colsToPrune);
                console.log("doing post prune");
                rowsToPrune.forEach(function (row) {
                    for (var i = 0; i < _this.model.gridw; i += 2) {
                        var coords = new CartesianCoords(i, row);
                        if (_this.model.get(coords))
                            _this.prune(coords);
                    }
                });
                colsToPrune.forEach(function (col) {
                    for (var j = 0; j < _this.model.gridh; j += 2) {
                        var coords = new CartesianCoords(col, j);
                        if (_this.model.get(coords))
                            _this.prune(coords);
                    }
                });
            };
            colsToRemove.forEach(function (col, i) {
                if (_this.model.gridw > 2) {
                    if (rowsToRemove.length == 0 && i == colsToRemove.length - 1)
                        _this.fold(0 /* X */, col, doPostPrune);
                    else
                        _this.fold(0 /* X */, col);
                }
            });
            rowsToRemove.forEach(function (row, i) {
                if (_this.model.gridh > 2) {
                    if (i == rowsToRemove.length - 1)
                        _this.fold(1 /* Y */, row, doPostPrune);
                    else
                        _this.fold(1 /* Y */, row);
                }
            });
        }
    };
    SquareGame.prototype.prune = function (start) {
        var _this = this;
        console.log("pruning at " + start);
        var startTile = this.model.get(start);
        if (!startTile.isTangible())
            return;
        var targets = this.model.floodAcquire(start);
        if (targets.length >= startTile.value) {
            console.log("pruning " + targets.length + " tiles");
            var str = "";
            targets.forEach(function (t) {
                str += _this.model.get(t).value + " ";
            });
            console.log(str);
            this.view.pop(this.canvas, targets, function () {
                targets.forEach(function (t, i) {
                    _this.tracker.tiles[startTile.value]--;
                    _this.model.set(t, new Tile(1 /* EMPTY */, -1));
                });
                _this.gameState.lastCleared = startTile.value;
            });
        }
    };
    SquareGame.prototype.simpleSlide = function (from, direction, doneCallback, numSlides) {
        var _this = this;
        if (doneCallback === void 0) { doneCallback = null; }
        if (numSlides === void 0) { numSlides = 1; }
        var to = from.map(function (c) {
            return c.displace(direction, 0.5);
        });
        this.view.slide(this.canvas, from, to, function () {
            if (doneCallback != null)
                doneCallback();
            _this.model.recomputeAllBounds();
        });
    };
    SquareGame.prototype.spawn = function (tileLine, side) {
        var _this = this;
        var insertIndex = 0;
        var insertDim = 0 /* X */;
        var blockDist = new TSM.vec2([0, 0]);
        switch (side) {
            case 3 /* LEFT */:
                blockDist.x = 0.5;
                break;
            case 2 /* RIGHT */:
                blockDist.x = -0.5;
                insertIndex = this.model.gridw;
                break;
            case 0 /* TOP */:
                blockDist.y = 0.5;
                insertDim = 1 /* Y */;
                break;
            case 1 /* BOTTOM */:
                blockDist.y = -0.5;
                insertDim = 1 /* Y */;
                insertIndex = this.model.gridh;
                break;
        }
        var everything = [];
        this.model.grid.forEach(function (t, i) {
            if (t.isTangible())
                everything.push(_this.model.toCoords(i));
        });
        this.view.physicalSlide(this.canvas, everything, blockDist, function () {
            _this.model.insert(insertDim, insertIndex, tileLine);
            tileLine.forEach(function (t) {
                if (t.type == 2 /* REGULAR */)
                    _this.tracker.tiles[t.value]++;
            });
        });
    };
    SquareGame.prototype.fold = function (dim, index, doneCallback) {
        var _this = this;
        if (doneCallback === void 0) { doneCallback = null; }
        switch (dim) {
            case 0 /* X */:
                var left = [], right = [];
                for (var i = 0; i < this.model.size; i++) {
                    var c = this.model.toCoords(i);
                    if (c.x < index)
                        left.push(c);
                    else if (c.x > index)
                        right.push(c);
                }
                console.log("fold column " + index);
                this.view.melt(this.canvas, dim, index, function () {
                    if (left.length > 0)
                        _this.simpleSlide(left, 1 /* EAST */, function () {
                            if (right.length == 0) {
                                _this.model.remove(dim, index);
                                if (doneCallback != null)
                                    doneCallback();
                            }
                        });
                    if (right.length > 0)
                        _this.simpleSlide(right, 3 /* WEST */, function () {
                            _this.model.remove(dim, index);
                            if (doneCallback != null)
                                doneCallback();
                        });
                });
                break;
            case 1 /* Y */:
                var above = [], below = [];
                for (var i = 0; i < this.model.size; i++) {
                    var c = this.model.toCoords(i);
                    if (c.y < index)
                        above.push(c);
                    else if (c.y > index)
                        below.push(c);
                }
                console.log("fold row " + index);
                this.view.melt(this.canvas, dim, index, function () {
                    if (above.length > 0)
                        _this.simpleSlide(above, 2 /* SOUTH */, function () {
                            if (below.length == 0) {
                                _this.model.remove(dim, index);
                                if (doneCallback != null)
                                    doneCallback();
                            }
                        });
                    if (below.length > 0)
                        _this.simpleSlide(below, 0 /* NORTH */, function () {
                            _this.model.remove(dim, index);
                            if (doneCallback != null)
                                doneCallback();
                        });
                });
                break;
        }
    };
    SquareGame.prototype.debugPrintView = function () {
        return this.view.debugPrint();
    };
    SquareGame.prototype.debugPrintModel = function () {
        return this.model.debugPrint();
    };
    SquareGame.prototype.debugPrintModelForLevelEditor = function () {
        return this.model.debugPrintForLevelEditor();
    };
    SquareGame.prototype.connectModelSignals = function (model) {
        var _this = this;
        model.modelSignals.moved.add(function (from, to, doneCallback) {
            _this.view.queueAnimation([new AnimationData(0 /* SLIDE */, [_this.canvas, from, to, doneCallback])]);
        }, this.view);
        model.modelSignals.crushed.add(function (crusheeCoords, crushDirection, doneCallback) {
            _this.view.queueAnimation([new AnimationData(4 /* CRUSH */, [_this.canvas, crusheeCoords, crushDirection])]);
        }, this.view);
        model.modelSignals.deleted.add(function (prunedCoords) {
            _this.view.queueAnimation([new AnimationData(1 /* POP */, [_this.canvas, prunedCoords])]);
        }, this.view);
    };
    SquareGame.prototype.connectViewSignals = function (view) {
        view.viewSignals.allFinished.add(function () {
        }, this);
    };
    SquareGame.prototype.initHooksForDebugger = function (dbg) {
        this.gameSignals.updated.add(dbg.onGameUpdated, dbg);
    };
    SquareGame.prototype.enter = function () {
    };
    SquareGame.prototype.leave = function () {
    };
    return SquareGame;
})();
var Debugger = (function () {
    function Debugger(game) {
        this.game = game;
        this.gridHistory = [];
    }
    Debugger.prototype.init = function () {
        var _this = this;
        Events.bind(document, 'keystroke.s', function (e) {
            console.log(_this.game.debugPrintModelForLevelEditor());
        });
        Events.bind(document, 'keystroke.Meta+Z', function (e) {
            _this.undo();
        });
        Events.bind(document, 'keystroke.Ctrl+Z', function (e) {
            _this.undo();
        });
    };
    Debugger.prototype.onGameUpdated = function (grid, gridw, gridh) {
        this.gridHistory.push({ grid: JSON.parse(grid), gridw: gridw, gridh: gridh });
    };
    Debugger.prototype.buildDebuggerUI = function () {
        var _this = this;
        function onLoadUILayout() {
            var d = new DOMParser();
            console.log(d.parseFromString(this.responseText, "text/html"));
        }
        var req = new XMLHttpRequest();
        req.onload = onLoadUILayout;
        req.open("get", "debugger.html", true);
        req.send();
        var debugPane = document.createElement('div');
        debugPane.id = "debug-pane";
        var fileUploadUI = document.createElement('input');
        debugPane.appendChild(fileUploadUI);
        fileUploadUI.type = "file";
        fileUploadUI.name = "files[]";
        fileUploadUI.addEventListener('change', function (e) {
            var files = e.target.files;
            for (var i = 0, file; file = files[i]; i++) {
                console.log(file);
                if (!file.type.match('.*json.*')) {
                    continue;
                }
                var reader = new FileReader();
                reader.onload = (function (f) {
                    return function (ee) {
                        try {
                            var levels = JSON.parse(ee.target.result);
                        }
                        catch (e) {
                            console.log(e);
                        }
                        finally {
                        }
                    };
                })(file);
                reader.readAsText(file);
            }
        }, false);
        var restartButtonUI = document.createElement('button');
        debugPane.appendChild(restartButtonUI);
        restartButtonUI.type = "button";
        restartButtonUI.innerHTML = "restart";
        restartButtonUI.addEventListener("click", function (e) {
            _this.game.restart();
        });
        var container = qwery("#screen")[0];
        container.parentNode.appendChild(debugPane);
    };
    Debugger.prototype.undo = function () {
        this.gridHistory.pop();
        var state = this.gridHistory.pop();
        this.game.importGridState(state.grid, state.gridw, state.gridh, this.game.gameParams.level.maxGridw, this.game.gameParams.level.maxGridh);
    };
    Debugger.canInstantiate = function () {
        var hasFileAPISupport = window.File && window.FileReader && window.FileList && window.Blob;
        return hasFileAPISupport;
    };
    return Debugger;
})();
var game;
var dbg;
var branch = function () {
    if (document.cookie.match(/^(.*;)?tess_prototype_seen_splash=[^;]+(.*)?$/)) {
        location.href = "play.html";
    }
};
var gotoGame = function () {
    var expires = new Date();
    expires.setFullYear(expires.getFullYear() + 1);
    var cookie = "tess_prototype_seen_splash=true; path=/tessellations/; expires=" + expires.toUTCString();
    document.cookie = cookie;
    location.href = "play.html";
};
var init = function () {
    ui.push(new ModeSelect());
};
var game;
var dbg;
var ModeSelect = (function () {
    function ModeSelect() {
    }
    ModeSelect.prototype.enter = function () {
        if (document.cookie.match(/^(.*;)?tess_prototype_seen_tutorial=[^;]+(.*)?$/)) {
        }
        else {
        }
        this.startNewGame();
    };
    ModeSelect.prototype.startNewGame = function () {
        var g = new SquareGame(1 /* PUZZLE */, 1 /* SVG */, null);
        game = g;
        var gp = {
            level: {
                no: 0,
                next: 0,
                maxVal: 9,
                maxGridw: 10,
                maxGridh: 10,
                gridw: 5,
                gridh: 5,
                layout: null,
                respawnInterval: 5
            },
            gameType: 1 /* PUZZLE */,
            drawBackend: 1 /* SVG */
        };
        g.init(gp);
        ui.push(g);
        if (window.location.href.indexOf("prototype") != -1) {
            if (Debugger.canInstantiate()) {
                dbg = new Debugger(game);
                dbg.init();
                dbg.buildDebuggerUI();
            }
            else {
                console.log("no debugger support");
            }
            game.initHooksForDebugger(dbg);
        }
    };
    ModeSelect.prototype.leave = function () {
    };
    return ModeSelect;
})();
var Tutorial = (function () {
    function Tutorial() {
    }
    Tutorial.prototype.enter = function () {
    };
    Tutorial.prototype.leave = function () {
        var expires = new Date();
        expires.setFullYear(expires.getFullYear() + 1);
        var cookie = "tess_prototype_seen_tutorial=true; path=/tessellations/; expires=" + expires.toUTCString();
        document.cookie = cookie;
    };
    return Tutorial;
})();
