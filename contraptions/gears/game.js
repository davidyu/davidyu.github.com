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
var Debugger = (function () {
    function Debugger(game) {
        this.game = game;
        this.gridHistory = [];
    }
    Debugger.prototype.init = function () {
        var _this = this;
        Events.bind(document, 'keystroke.s', function (e) {
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
    CartesianCoords.displace = function (coords, direction) {
        return new CartesianCoords(coords.x + direction.x, coords.y + direction.y);
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
                deleted: new signals.Signal(),
                rotated: new signals.Signal()
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
        Square.prototype.move = function (target, direction) {
            var _this = this;
            console.log("moving " + target);
            var from = this.floodAcquire(target);
            var to = from.map(function (c) {
                return _this.getTileBoundInDirection(c, direction);
            });
            var fromTiles = from.map(function (f) {
                return _this.get(f);
            });
            console.log("before move:");
            console.log(this.debugPrint());
            to.forEach(function (t, i) {
                if (!to.some(function (h) {
                    return h.equals(from[i]);
                })) {
                    _this.set(from[i], new Tile(1 /* EMPTY */, -1));
                }
                _this.set(t, new Tile(fromTiles[i].type, fromTiles[i].value));
            });
            this.modelSignals.moved.dispatch(from, to);
            console.log("after move:");
            console.log(this.debugPrint());
            var skipPrune = false;
            {
                var patty = to.map(function (c) {
                    return c.displace(direction);
                });
                var bun = patty.map(function (c) {
                    return c.displace(direction);
                });
                this.checkCollision(from, patty).forEach(function (col, i) {
                    if (col && _this.get(patty[i]).type != 0 /* OUT_OF_BOUNDS */ && _this.get(patty[i]).isTangible()) {
                        if (_this.get(patty[i]).value < _this.get(to[i]).value) {
                            if (_this.get(bun[i]).type == 0 /* OUT_OF_BOUNDS */ || _this.get(bun[i]).value > _this.get(patty[i]).value) {
                                skipPrune = true;
                                _this.set(patty[i], new Tile(1 /* EMPTY */, -1));
                                _this.modelSignals.crushed.dispatch(patty[i], direction);
                                _this.recomputeAllBounds();
                                _this.move(to[i], direction);
                            }
                            else {
                                _this.move(patty[i], direction);
                            }
                        }
                    }
                });
                if (!skipPrune)
                    this.prune(to[0]);
                this.recomputeAllBounds();
            }
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
            this.genHalf(0, Math.floor(this.gridh / 2), min, max);
            this.genHalf(Math.ceil(this.gridh / 2), this.gridh, min, max);
            for (var y = 0; y < this.gridh; y++) {
                for (var x = 0; x < this.gridw; x++) {
                    var coords = new CartesianCoords(x, y);
                    var t = this.get(coords);
                    if (this.floodAcquire(coords).length == t.value) {
                        do {
                            this.set(coords, new Tile(1 /* EMPTY */, -1));
                            var newCoords = new CartesianCoords(Math.round(Math.random() * this.gridw), Math.round(Math.random() * this.gridh));
                            while (this.get(newCoords).isTangible()) {
                                newCoords = new CartesianCoords(Math.round(Math.random() * this.gridw), Math.round(Math.random() * this.gridh));
                            }
                            this.set(newCoords, t);
                            var canPop = this.floodAcquire(newCoords).length == t.value;
                        } while (canPop);
                    }
                }
            }
            this.recomputeAllBounds();
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
var HexDirection;
(function (HexDirection) {
    HexDirection[HexDirection["NE"] = 0] = "NE";
    HexDirection[HexDirection["E"] = 1] = "E";
    HexDirection[HexDirection["SE"] = 2] = "SE";
    HexDirection[HexDirection["SW"] = 3] = "SW";
    HexDirection[HexDirection["W"] = 4] = "W";
    HexDirection[HexDirection["NW"] = 5] = "NW";
})(HexDirection || (HexDirection = {}));
var RotateDirection;
(function (RotateDirection) {
    RotateDirection[RotateDirection["LEFT"] = 0] = "LEFT";
    RotateDirection[RotateDirection["RIGHT"] = 1] = "RIGHT";
})(RotateDirection || (RotateDirection = {}));
var AxialCoords = (function () {
    function AxialCoords(q, r) {
        this.q = q;
        this.r = r;
    }
    AxialCoords.prototype.displace = function (arg, magnitude) {
        if (magnitude === void 0) { magnitude = 1; }
        if (arg instanceof AxialCoords) {
            var delta = arg;
            return new AxialCoords(this.q + delta.q, this.r + delta.r);
        }
        else {
            switch (arg) {
                case 0 /* NE */:
                    return new AxialCoords(this.q + magnitude, this.r - magnitude);
                    break;
                case 1 /* E */:
                    return new AxialCoords(this.q + magnitude, this.r);
                    break;
                case 2 /* SE */:
                    return new AxialCoords(this.q, this.r + magnitude);
                    break;
                case 3 /* SW */:
                    return new AxialCoords(this.q - magnitude, this.r + magnitude);
                    break;
                case 4 /* W */:
                    return new AxialCoords(this.q - magnitude, this.r);
                    break;
                case 5 /* NW */:
                    return new AxialCoords(this.q, this.r - magnitude);
                    break;
            }
        }
    };
    AxialCoords.prototype.rotate = function (direction) {
        switch (direction) {
            case 0 /* LEFT */:
                return new AxialCoords(-this.r, this.r + this.q);
                break;
            case 1 /* RIGHT */:
                return new AxialCoords(this.r + this.q, -this.q);
                break;
        }
    };
    AxialCoords.displace = function (coords, direction) {
        return new AxialCoords(coords.q + direction.q, coords.r + direction.r);
    };
    AxialCoords.prototype.toString = function () {
        return "( " + this.q + ", " + this.r + " ) => ( " + this.q + ", " + (0 - (this.q + this.r)) + ", " + this.r + " )";
    };
    return AxialCoords;
})();
var Model;
(function (Model) {
    var Hex = (function () {
        function Hex(gridr, DefaultTile, OutOfBoundsTile) {
            this.grid = [];
            this.gridr = gridr;
            this.size = this.diameter() * this.diameter();
            this.outOfBoundsTile = OutOfBoundsTile;
            for (var r = -this.gridr; r <= this.gridr; r++) {
                for (var q = -this.gridr; q <= this.gridr; q++) {
                    if (Math.abs(r + q) > this.gridr)
                        this.grid.push(OutOfBoundsTile);
                    else
                        this.grid.push(DefaultTile);
                }
            }
            this.signals = {
                moved: new signals.Signal(),
                crushed: new signals.Signal(),
                deleted: new signals.Signal(),
                rotated: new signals.Signal()
            };
        }
        Hex.prototype.diameter = function () {
            return (this.gridr * 2 + 1);
        };
        Hex.prototype.toFlat = function (q, r) {
            var x = r + this.gridr;
            var z = q + this.gridr;
            return x + z * (this.diameter());
        };
        Hex.prototype.toCoords = function (i) {
            var x = i % this.diameter();
            var z = Math.floor(i / this.diameter());
            return new AxialCoords(x - this.gridr, z - this.gridr);
        };
        Hex.prototype.get = function (c) {
            return Math.abs(c.q) <= this.gridr && Math.abs(c.r) <= this.gridr ? this.grid[this.toFlat(c.q, c.r)] : this.outOfBoundsTile;
        };
        Hex.prototype.getFlat = function (i) {
            return i >= 0 && i < this.grid.length ? this.grid[i] : this.outOfBoundsTile;
        };
        Hex.prototype.set = function (c, tile) {
            if (Math.abs(c.q) <= this.gridr && Math.abs(c.r) <= this.gridr) {
                this.grid[this.toFlat(c.q, c.r)] = tile;
            }
        };
        Hex.prototype.setFlat = function (i, tile) {
            if (i >= 0 && i < this.grid.length) {
                this.grid[i] = tile;
            }
        };
        Hex.prototype.isEmpty = function (arg) {
            if (typeof (arg) == "number") {
                return this.getFlat(arg).type == 1 /* EMPTY */;
            }
            else {
                return this.get(arg).type == 1 /* EMPTY */;
            }
        };
        Hex.prototype.rotate = function (direction) {
            var counter = direction == 0 /* LEFT */ ? 1 /* RIGHT */ : 0 /* LEFT */;
            for (var i = 1; i < this.gridr; i++) {
                for (var j = 0; j < i; j++) {
                    var startCoords = this.center().displace(5 /* NW */, i).displace(1 /* E */, j);
                    var startTile = this.get(startCoords);
                    var iterations = 0;
                    var coords = Utils.deepCopy(startCoords);
                    while (iterations < 5) {
                        var prev = coords.rotate(counter);
                        this.set(coords, this.get(prev));
                        iterations++;
                        coords = coords.rotate(counter);
                    }
                    this.set(coords, startTile);
                }
            }
            this.signals.rotated.dispatch(direction);
        };
        Hex.prototype.center = function () {
            return new AxialCoords(0, 0);
        };
        Hex.prototype.floodAcquire = function (start) {
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
            if (this.get(new AxialCoords(start.q, start.r)) != tile)
                return [];
            Q.push(start);
            while (Q.length > 0) {
                var n = Q.shift();
                if (this.get(n).value == tile.value && this.get(n).type == tile.type && !marked.get(n)) {
                    var east = new AxialCoords(n.q, n.r);
                    var west = new AxialCoords(n.q, n.r);
                    while (this.get(east.displace(1 /* E */)).value == tile.value && this.get(east.displace(1 /* E */)).type == tile.type) {
                        east = east.displace(1 /* E */);
                    }
                    while (this.get(west.displace(4 /* W */)).value == tile.value && this.get(west.displace(4 /* W */)).type == tile.type) {
                        west = west.displace(4 /* W */);
                    }
                    for (var r = east.r; r <= west.r; r++) {
                        var nn = new AxialCoords(east.q, r);
                        marked.set(nn);
                        cluster.push(nn);
                        var nw = nn.displace(5 /* NW */);
                        var ne = nn.displace(0 /* NE */);
                        var sw = nn.displace(3 /* SW */);
                        var se = nn.displace(2 /* SE */);
                        if (this.get(nw).value == tile.value && this.get(nw).type == tile.type)
                            Q.push(nw);
                        if (this.get(ne).value == tile.value && this.get(ne).type == tile.type)
                            Q.push(ne);
                        if (this.get(sw).value == tile.value && this.get(sw).type == tile.type)
                            Q.push(sw);
                        if (this.get(se).value == tile.value && this.get(se).type == tile.type)
                            Q.push(se);
                    }
                }
            }
            return cluster;
        };
        Hex.prototype.getTileArray = function () {
            return this.grid;
        };
        Hex.prototype.computeDistance = function (a, b) {
            return (Math.abs(a.q - b.q) + Math.abs(a.q + a.r - b.q - b.r) + Math.abs(a.r - b.r)) / 2;
        };
        Hex.prototype.procGenGrid = function (min, max, radius) {
            var origin = new AxialCoords(0, 0);
            for (var r = -radius; r <= radius; r++) {
                for (var q = -radius; q <= radius; q++) {
                    if (Math.abs(r + q) <= radius) {
                        var coords = new AxialCoords(q, r);
                        var val = Math.round(Math.random() * (max - min) + min);
                        this.set(coords, new Tile(2 /* REGULAR */, val));
                    }
                }
            }
        };
        Hex.prototype.debugPrint = function (plain) {
            if (plain === void 0) { plain = false; }
            if (plain) {
                var str = "";
                for (var r = -this.gridr; r <= this.gridr; r++) {
                    for (var q = -this.gridr; q <= this.gridr; q++) {
                        if (Math.abs(r + q) > this.gridr) {
                            str += "-";
                        }
                        else {
                            var t = this.get(new AxialCoords(q, r));
                            str += t.isTangible() ? t.value.toString() : "-";
                        }
                        str += " ";
                    }
                    str += "\n";
                }
                return str;
            }
            else {
                var indentations = 0;
                var str = "";
                for (var r = -this.gridr; r <= this.gridr; r++) {
                    for (var q = -this.gridr; q <= this.gridr; q++) {
                        if (Math.abs(r + q) > this.gridr) {
                            str += " ";
                        }
                        else {
                            var t = this.get(new AxialCoords(q, r));
                            str += t.isTangible() ? t.value.toString() : " ";
                        }
                        str += " ";
                    }
                    str += "\n";
                    indentations++;
                    for (var i = 0; i < indentations; i++) {
                        str += " ";
                    }
                }
                return str;
            }
        };
        return Hex;
    })();
    Model.Hex = Hex;
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
    AnimationType[AnimationType["ROTATE"] = 6] = "ROTATE";
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
        this.shadowFromTile = function (t) {
            if (t.type == 1 /* EMPTY */) {
                return this.scale[t.type](t.value / 9).hex();
            }
            else {
                return this.scale[t.type](t.value / 9).darken(10).hex();
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
            var cellw = 0, cellh = 0;
            var margin = 20;
            switch (this.drawBackend) {
                case 0 /* CANVAS */:
                    break;
                case 1 /* SVG */:
                    cellw = Math.floor((playableRect.w - margin) / this.model.maxgridw), cellh = Math.floor((playableRect.h - margin) / this.model.maxgridh);
                    break;
            }
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
                var text = cell.plain(e.isTangible() ? e.value.toString() : "");
                text.attr({ 'fill': this.colorizer.foregroundFromColor(this.colorizer.fromTile(e)), 'font-size': cellw / 4, 'cursor': 'inherit' }).transform({ x: -text.attr('font-size') / 4, y: text.attr('font-size') / 4 });
                cell.coords = new CartesianCoords(x, y);
                cell.rect = rect;
                cell.text = text;
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
                this.bg.attr({ "x": Math.floor(this.playableRect.x + this.gridOffset.x) + stroke / 2, "y": Math.floor(this.playableRect.y + this.gridOffset.y) + stroke / 2, "width": gridDim.x - stroke / 2, "height": gridDim.y - stroke / 2, "fill": '#000' });
            }
            for (var y = 0; y < this.model.gridh; y++) {
                for (var x = 0; x < this.model.gridw; x++) {
                    var e = this.model.get(new CartesianCoords(x, y));
                    if (this.emptyBackings[this.model.toFlat(x, y)] != null)
                        this.emptyBackings[this.model.toFlat(x, y)].remove();
                    this.emptyBackings[this.model.toFlat(x, y)] = this.drawEmptyBacking(canvas, x, y, e);
                }
            }
            for (var y = 0; y < this.model.gridh; y++) {
                for (var x = 0; x < this.model.gridw; x++) {
                    var e = this.model.get(new CartesianCoords(x, y));
                    var offs = new TSM.vec2([0, 0]);
                    if (this.cells[this.model.toFlat(x, y)] != null) {
                        this.cells[this.model.toFlat(x, y)].remove();
                    }
                    this.cells[this.model.toFlat(x, y)] = this.drawTile(canvas, x, y, e, offs);
                }
            }
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
var View;
(function (View) {
    var HexDim = (function () {
        function HexDim() {
        }
        return HexDim;
    })();
    var HexView = (function () {
        function HexView(hex, backend, canvas, gameSignals) {
            this.model = hex;
            this.animQ = [];
            this.animating = false;
            this.cellSz = this.computeCellSize(View.computePlayableRect());
            this.colorizer = new Colorizer();
            this.concurrentAnimations = 0;
            this.drawBackend = backend;
            this.playableRect = View.computePlayableRect();
            this.signals = {
                animationFinished: new signals.Signal(),
                allFinished: new signals.Signal()
            };
            this.signals.animationFinished.add(this.animationFinished, this);
            this.debugVisualizations = false;
        }
        HexView.prototype.getCellSize = function () {
            return this.cellSz;
        };
        HexView.prototype.computeCellSize = function (playableRect) {
            var margin = 20;
            var minDim = Math.min(playableRect.w, playableRect.h);
            var cellr = Math.floor((minDim - margin) / this.model.diameter() / 2);
            var cellR = cellr / Math.sin(60 * Math.PI / 180);
            var d = { r: cellr, R: cellR };
            return d;
        };
        HexView.prototype.toAbsX = function (q) {
            return q + this.model.gridr;
        };
        HexView.prototype.toAbsY = function (r) {
            return r + this.model.gridr;
        };
        HexView.prototype.drawTile = function (canvas, q, r, e) {
            var size = this.getCellSize().r * 2;
            var radius = this.getCellSize().R;
            var yd = radius * Math.cos(60 * Math.PI / 180);
            var xd = radius * Math.sin(60 * Math.PI / 180);
            var yOffset = radius;
            var xOffset = xd;
            var x = this.toAbsX(q);
            var y = this.toAbsY(r);
            var centerX = size * x + xd * r + xOffset;
            var centerY = (yd + radius) * y + yOffset;
            var cell = canvas.group().transform({ x: centerX, y: centerY });
            var pts = [new Vec2(0, -radius), new Vec2(xd, -yd), new Vec2(xd, yd), new Vec2(0, radius), new Vec2(-xd, yd), new Vec2(-xd, -yd)];
            var ptstr = pts.reduce(function (p1, p2, i, v) {
                return p1.toString() + " " + p2.toString();
            }, "");
            var hex = cell.polygon(ptstr);
            if (this.debugVisualizations) {
                var text = cell.plain("(" + q + ", " + r + "): " + e.value).transform({ x: -30, y: 0 });
            }
            hex.attr({ 'fill': this.colorizer.fromTile(e), 'stroke': '#fff', 'stroke-width': 2 });
            cell.coords = new AxialCoords(q, r);
            cell.hex = hex;
            return cell;
        };
        HexView.prototype.drawEmptyBacking = function (canvas, q, r, e) {
            var size = this.getCellSize().r * 2;
            var radius = this.getCellSize().R;
            var yd = radius * Math.cos(60 * Math.PI / 180);
            var xd = radius * Math.sin(60 * Math.PI / 180);
            var yOffset = radius;
            var xOffset = xd;
            var x = this.toAbsX(q);
            var y = this.toAbsY(r);
            var centerX = size * x + xd * r + xOffset;
            var centerY = (yd + radius) * y + yOffset;
            var cell;
            if (this.drawBackend == 1 /* SVG */) {
                cell = canvas.group().transform({ x: centerX, y: centerY });
                var pts = [new Vec2(0, -radius), new Vec2(xd, -yd), new Vec2(xd, yd), new Vec2(0, radius), new Vec2(-xd, yd), new Vec2(-xd, -yd)];
                var ptstr = pts.reduce(function (p1, p2, i, v) {
                    return p1.toString() + " " + p2.toString();
                }, "");
                var hex = cell.polygon(ptstr).attr({ 'fill': '#fff' });
            }
            return cell;
        };
        HexView.prototype.resetView = function (canvas) {
            if (this.model == null)
                return;
            if (this.emptyBackings == null)
                this.emptyBackings = [];
            this.cells = [];
            canvas.clear();
            this.topGroup = canvas.group();
            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(new AxialCoords(q, r));
                    if (this.emptyBackings[this.model.toFlat(q, r)] != null)
                        this.emptyBackings[this.model.toFlat(q, r)].remove();
                    this.emptyBackings[this.model.toFlat(q, r)] = this.drawEmptyBacking(this.topGroup, q, r, e);
                }
            }
            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(new AxialCoords(q, r));
                    if (e.type == 2 /* REGULAR */) {
                        this.cells[this.model.toFlat(q, r)] = this.drawTile(this.topGroup, q, r, e);
                    }
                }
            }
        };
        HexView.prototype.getViewElements = function () {
            return this.cells;
        };
        HexView.prototype.getViewElement = function (c) {
            return Math.abs(c.q) <= this.model.gridr && Math.abs(c.r) <= this.model.gridr ? this.cells[this.model.toFlat(c.q, c.r)] : null;
        };
        HexView.prototype.setViewElement = function (c, e) {
            if (Math.abs(c.q) <= this.model.gridr && Math.abs(c.r) <= this.model.gridr) {
                this.cells[this.model.toFlat(c.q, c.r)] = e;
            }
        };
        HexView.prototype.rotate = function (canvas, direction, doneCallback) {
            var _this = this;
            var target = canvas.group();
            target.animate(150, '>', 0).during(function (t) {
                var sign = direction == 0 /* LEFT */ ? 1 : -1;
                var deg = sign * 60 * SVG.easing.cubicOut(t);
                _this.topGroup.rotate(deg);
            }).after(function () {
                _this.resetView(canvas);
                _this.signals.animationFinished.dispatch();
                if (doneCallback != null)
                    doneCallback();
                target.remove();
            });
        };
        HexView.prototype.queueAnimation = function (anims) {
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
        HexView.prototype.playAnimations = function (anims) {
            var _this = this;
            this.animating = true;
            anims.forEach(function (a) {
                _this.concurrentAnimations++;
                switch (a.type) {
                    case 0 /* SLIDE */: break;
                    case 1 /* POP */: break;
                    case 2 /* MATERIALIZE */: break;
                    case 3 /* MELT */: break;
                    case 4 /* CRUSH */: break;
                    case 6 /* ROTATE */:
                        _this.rotate.apply(_this, a.params);
                        break;
                }
            });
        };
        HexView.prototype.animationFinished = function () {
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
                this.signals.allFinished.dispatch();
            }
        };
        return HexView;
    })();
    View.HexView = HexView;
})(View || (View = {}));
var HexGameState = (function () {
    function HexGameState() {
    }
    return HexGameState;
})();
var HexGame = (function () {
    function HexGame(gp) {
        this.params = gp;
        this.drawBackend = gp.drawBackend;
        this.canvas = View.buildCanvas(gp.drawBackend);
        this.model = new Model.Hex(5, new Tile(1 /* EMPTY */, 0), new Tile(0 /* OUT_OF_BOUNDS */, -1));
        this.view = new View.HexView(this.model, gp.drawBackend, this.canvas, null);
        this.inputQ = [];
        this.maxInputQSlots = 5;
    }
    HexGame.prototype.connectSignals = function (model, view) {
        var _this = this;
        model.signals.rotated.add(function (direction) {
            _this.view.queueAnimation([new AnimationData(6 /* ROTATE */, [_this.canvas, direction])]);
        }, this.view);
        view.signals.allFinished.add(function () {
            if (_this.inputQ.length > 0) {
                var dir = _this.inputQ.shift();
                _this.model.rotate(dir);
            }
            else {
                _this.updateViewAndController();
            }
        }, this.view);
    };
    HexGame.prototype.init = function () {
        var _this = this;
        if (this.params.level == null) {
            this.model.procGenGrid(1, 4, 3);
        }
        View.resetCanvas(this.canvas, this.drawBackend);
        this.updateViewAndController();
        Events.bind(document, 'keystroke.left', function (e) { return _this.rotateLeft(e); });
        Events.bind(document, 'keystroke.right', function (e) { return _this.rotateRight(e); });
        this.connectSignals(this.model, this.view);
    };
    HexGame.prototype.rotateLeft = function (e) {
        if (!this.view.animating) {
            this.model.rotate(0 /* LEFT */);
        }
        else if (this.inputQ.length < this.maxInputQSlots) {
            this.inputQ.push(0 /* LEFT */);
        }
    };
    HexGame.prototype.rotateRight = function (e) {
        if (!this.view.animating) {
            this.model.rotate(1 /* RIGHT */);
        }
        else if (this.inputQ.length < this.maxInputQSlots) {
            this.inputQ.push(1 /* RIGHT */);
        }
    };
    HexGame.prototype.restart = function () {
    };
    HexGame.prototype.update = function () {
    };
    HexGame.prototype.timedUpdate = function () {
    };
    HexGame.prototype.updateViewAndController = function () {
        this.view.resetView(this.canvas);
        this.extendUIFromView();
    };
    HexGame.prototype.extendUIFromView = function () {
        var _this = this;
        var pivots = [new AxialCoords(0, 0)];
        pivots.forEach(function (p) {
            var elem = _this.view.getViewElement(p);
            elem.click(function () {
                _this.rotateLeft(null);
            });
        });
    };
    HexGame.prototype.importGridState = function (state, gridw, gridh, maxgridw, maxgridh) {
    };
    HexGame.prototype.initHooksForDebugger = function (dbg) {
    };
    HexGame.prototype.enter = function () {
        this.init();
    };
    HexGame.prototype.leave = function () {
    };
    HexGame.prototype.addSystem = function (s) {
    };
    return HexGame;
})();
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
var Entity = (function () {
    function Entity() {
    }
    return Entity;
})();
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
        var gp = {
            level: null,
            gameType: 0 /* SURVIVAL */,
            drawBackend: 1 /* SVG */
        };
        var g = new HexGame(gp);
        game = g;
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
